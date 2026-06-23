"""
Checkpoint-compatible GTCA implementation (model-agnostic).

- ParseTreeEncoder: builds per-height chunk memory from span-aligned tree nodes.
- ParseTreeCrossAttention: gated cross-attention from token states to chunk memory with a causal chunk mask.
- GTCAModel: wraps a decoder-only causal LM and injects cross-attention into each self-attention layer via wrappers.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads for grouped-query attention.

    hidden_states: (B, n_kv_heads, S, head_dim)
    returns: (B, n_kv_heads * n_rep, S, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    b, n_kv, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)


def _get_base_model(m: Any) -> Any:
    if hasattr(m, "get_base_model"):
        try:
            return m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model"):
        return m.base_model
    return m


def _get_decoder_layers(base_model: Any) -> List[Any]:
    """
    Try to obtain the list of decoder layers for common decoder-only HF architectures.
    """
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return list(base_model.model.layers)
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        return list(base_model.transformer.h)
    raise ValueError("Could not locate decoder layers (expected model.layers or transformer.h).")


def _get_self_attention(layer: Any) -> Any:
    for name in ("self_attn", "attn", "attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError("Could not locate self-attention module in a decoder layer.")


def _attention_shapes(attn: Any, config: Any) -> Tuple[int, int, int]:
    """
    Return (num_heads, num_kv_heads, head_dim).
    """
    num_heads = getattr(attn, "num_heads", None) or getattr(config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("Could not infer num_heads from attention module or config.")
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer head_dim.")
        head_dim = hidden_size // int(num_heads)

    num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(config, "num_key_value_heads", None) or int(num_heads)
    return int(num_heads), int(num_kv_heads), int(head_dim)


def _mean_pool(token_states: torch.Tensor, idx: Sequence[int]) -> torch.Tensor:
    """
    token_states: (T, D)
    idx: token indices
    returns: (D,)
    """
    if not idx:
        return token_states.new_zeros((token_states.shape[-1],))
    sel = token_states.index_select(0, torch.tensor(idx, device=token_states.device))
    return sel.mean(dim=0)


def _compute_max_depth(node: Dict[str, Any], depth: int = 0) -> int:
    children = node.get("children") or []
    if not children:
        return depth
    return max(_compute_max_depth(ch, depth + 1) for ch in children)


def _bfs_nodes_with_depth(root: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int]]:
    out: List[Tuple[Dict[str, Any], int]] = []
    queue: List[Tuple[Dict[str, Any], int]] = [(root, 0)]
    while queue:
        node, d = queue.pop(0)
        out.append((node, d))
        children = node.get("children") or []
        for ch in children:
            queue.append((ch, d + 1))
    return out


class ParseTreeEncoder(nn.Module):
    """
    Build chunk memory by mean-pooling token embeddings over span-aligned nodes, then applying
    height-specific linear projections and LayerNorm.

    A node's height is defined as: height = D - depth(node), where D is the max depth of leaves.
    Leaves thus have height 0.
    """

    def __init__(self, hidden_dim: int, num_layers: int, max_chunks_per_height: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_chunks_per_height = int(max_chunks_per_height)

        # Height-specific projections (indexed by height, clamped to [0, num_layers-1]).
        self.proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        token_embeddings: torch.Tensor,  # (B, T, D)
        tree_structures: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Returns:
          chunk_memory_by_height: height -> (B, M, D)
          right_bounds_by_height: height -> (B, M) where each entry is the max token index for that chunk
        """
        bsz, t, d = token_embeddings.shape
        assert d == self.hidden_dim

        # Per-sample extraction
        per_sample_chunks: List[Dict[int, List[torch.Tensor]]] = []
        per_sample_bounds: List[Dict[int, List[int]]] = []

        for b in range(bsz):
            root = tree_structures[b]
            max_depth = _compute_max_depth(root, depth=0)
            bfs_nodes = _bfs_nodes_with_depth(root)

            chunks_by_h: Dict[int, List[torch.Tensor]] = {}
            bounds_by_h: Dict[int, List[int]] = {}

            for node, depth in bfs_nodes:
                tok_idx = node.get("token_indices") or []
                if not tok_idx:
                    continue
                # Height definition: leaves => 0, root => max_depth
                h = max_depth - depth
                if h < 0:
                    continue
                # Clamp height to available projection range
                h_clamped = min(max(h, 0), self.num_layers - 1)

                # Enforce K per height (left-to-right BFS order)
                if len(chunks_by_h.get(h_clamped, [])) >= self.max_chunks_per_height:
                    continue

                pooled = _mean_pool(token_embeddings[b], tok_idx)  # (D,)
                proj = self.proj[h_clamped](pooled)
                proj = self.norm[h_clamped](proj)
                chunks_by_h.setdefault(h_clamped, []).append(proj)

                rb = int(max(tok_idx))
                bounds_by_h.setdefault(h_clamped, []).append(rb)

            per_sample_chunks.append(chunks_by_h)
            per_sample_bounds.append(bounds_by_h)

        # Collate to (B, M, D) per height
        chunk_memory_by_h: Dict[int, torch.Tensor] = {}
        right_bounds_by_h: Dict[int, torch.Tensor] = {}

        all_heights = set()
        for dct in per_sample_chunks:
            all_heights.update(dct.keys())
        for h in sorted(all_heights):
            max_m = max(len(per_sample_chunks[b].get(h, [])) for b in range(bsz))
            if max_m == 0:
                continue

            mem = token_embeddings.new_zeros((bsz, max_m, d))
            bounds = torch.full((bsz, max_m), -1, device=token_embeddings.device, dtype=torch.long)

            for b in range(bsz):
                chunks = per_sample_chunks[b].get(h, [])
                rbs = per_sample_bounds[b].get(h, [])
                if not chunks:
                    continue
                m = len(chunks)
                mem[b, :m, :] = torch.stack(chunks, dim=0)
                bounds[b, :m] = torch.tensor(rbs, device=token_embeddings.device, dtype=torch.long)

            chunk_memory_by_h[h] = mem
            right_bounds_by_h[h] = bounds

        return chunk_memory_by_h, right_bounds_by_h


class ParseTreeCrossAttention(nn.Module):
    """
    Gated cross-attention from token states to chunk memory.

    - Queries/Gates computed from token states.
    - Keys/Values computed from chunk memory.
    - Causal chunk mask blocks attending to chunks whose right boundary is in the future.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.g_proj = nn.Linear(hidden_dim, num_heads, bias=True)  # one logit per head
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        token_states: torch.Tensor,              # (B, T, D)
        chunk_states: Optional[torch.Tensor],    # (B, M, D) or None
        chunk_right_bounds: Optional[torch.Tensor],  # (B, M) or None
        token_update_mask: Optional[torch.Tensor],   # (B, T) bool or float, 1 means allow update
        alpha: float,
    ) -> torch.Tensor:
        if chunk_states is None or chunk_states.numel() == 0:
            return token_states.new_zeros(token_states.shape)

        bsz, t, d = token_states.shape
        _, m, _ = chunk_states.shape

        q = self.q_proj(token_states).view(bsz, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(chunk_states).view(bsz, m, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, M, Dh)
        v = self.v_proj(chunk_states).view(bsz, m, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, M, Dh)

        n_rep = self.num_heads // self.num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}).")
        k = repeat_kv(k, n_rep)  # (B, H, M, Dh)
        v = repeat_kv(v, n_rep)  # (B, H, M, Dh)

        attn_mask = None
        if chunk_right_bounds is not None:
            # Mask out chunks with right bound > current token position
            # chunk_right_bounds: (B, M)
            token_pos = torch.arange(t, device=token_states.device, dtype=torch.long).view(1, 1, t, 1)
            rb = chunk_right_bounds.view(bsz, 1, 1, m)
            # Mask padded chunks (rb < 0) and future chunks (rb > token_pos)
            attn_mask = (rb < 0) | (rb > token_pos)  # bool, True = mask out

        # scaled_dot_product_attention: returns (B, H, T, Dh)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        gate = torch.sigmoid(self.g_proj(token_states))  # (B, T, H)
        gate = gate.transpose(1, 2).unsqueeze(-1)        # (B, H, T, 1)
        attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t, self.num_heads * self.head_dim)
        delta = self.o_proj(attn_out)  # (B, T, D)

        if token_update_mask is not None:
            if token_update_mask.dtype != delta.dtype:
                token_update_mask = token_update_mask.to(dtype=delta.dtype)
            delta = delta * token_update_mask.unsqueeze(-1)

        return delta * float(alpha)


@dataclass
class CrossAttnWrapperHelper:
    cross_attn: ParseTreeCrossAttention
    alpha: float = 0.0
    chunk_tensor: Optional[torch.Tensor] = None
    chunk_right_bounds: Optional[torch.Tensor] = None
    token_update_mask: Optional[torch.Tensor] = None


class GTCAModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        lora_config: Optional[LoraConfig],
        tokenizer: Any,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_dropout: float = 0.0,
        max_chunks_per_height: int = 64,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        if lora_config is not None:
            self.backbone = get_peft_model(self.backbone, lora_config)

        base = _get_base_model(self.backbone)
        self.config = getattr(base, "config", None)
        if self.config is None:
            raise ValueError("Backbone model has no config.")

        hidden_size = int(getattr(self.config, "hidden_size"))
        num_layers = int(getattr(self.config, "num_hidden_layers"))

        self.parse_tree_encoder = ParseTreeEncoder(
            hidden_dim=hidden_size,
            num_layers=num_layers,
            max_chunks_per_height=max_chunks_per_height,
        )

        self.cross_attn_layers = nn.ModuleList()
        self._helpers: Dict[int, CrossAttnWrapperHelper] = {}

        # Wrap attention modules (one GTCA module per layer)
        self._attach_wrappers(attn_dropout=attn_dropout)

    def _attach_wrappers(self, attn_dropout: float) -> None:
        base = _get_base_model(self.backbone)
        layers = _get_decoder_layers(base)

        for layer_idx, layer in enumerate(layers):
            attn = _get_self_attention(layer)
            num_heads, num_kv_heads, head_dim = _attention_shapes(attn, self.config)
            hidden_dim = int(getattr(self.config, "hidden_size"))

            ca = ParseTreeCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dropout=attn_dropout,
            )
            self.cross_attn_layers.append(ca)
            helper = CrossAttnWrapperHelper(cross_attn=ca)
            self._helpers[layer_idx] = helper

            original_forward = attn.forward

            def make_wrapped_forward(of, h: CrossAttnWrapperHelper):
                def wrapped_forward(*args, **kwargs):
                    # Many HF attention modules take hidden_states as first arg.
                    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    out = of(*args, **kwargs)

                    if hidden_states is None:
                        return out

                    if isinstance(out, tuple):
                        attn_out = out[0]
                        rest = out[1:]
                    else:
                        attn_out = out
                        rest = None

                    delta = h.cross_attn(
                        token_states=hidden_states,
                        chunk_states=h.chunk_tensor,
                        chunk_right_bounds=h.chunk_right_bounds,
                        token_update_mask=h.token_update_mask,
                        alpha=h.alpha,
                    )
                    attn_out = attn_out + delta

                    if rest is None:
                        return attn_out
                    return (attn_out,) + rest

                return wrapped_forward

            attn.forward = make_wrapped_forward(original_forward, helper)

    @torch.no_grad()
    def set_alpha(self, alpha: float) -> None:
        for h in self._helpers.values():
            h.alpha = float(alpha)

    def set_chunk_representations(
        self,
        input_ids: torch.Tensor,          # (B, T)
        attention_mask: torch.Tensor,     # (B, T)
        parsed_list: Optional[List[Optional[Dict[str, Any]]]],
    ) -> None:
        """
        Prepare per-layer chunk memory and token update masks for the current batch.
        """
        bsz, t = input_ids.shape
        device = input_ids.device

        # Clear helpers by default
        for h in self._helpers.values():
            h.chunk_tensor = None
            h.chunk_right_bounds = None
            h.token_update_mask = None

        if parsed_list is None or all(p is None for p in parsed_list):
            return

        # Compute token embeddings (E_i in the paper)
        embed_layer = self.backbone.get_input_embeddings()
        token_embeds = embed_layer(input_ids)  # (B, T, D)

        # Remap indices from unpadded to padded positions (left padding)
        remapped_trees: List[Dict[str, Any]] = []
        update_masks = torch.zeros((bsz, t), device=device, dtype=torch.float32)

        for b in range(bsz):
            parsed = parsed_list[b]
            if parsed is None:
                remapped_trees.append({"token_indices": [], "children": []})
                continue

            unpadded_len = int(attention_mask[b].sum().item())
            pad_len = t - unpadded_len

            def remap_node(node: Dict[str, Any]) -> Dict[str, Any]:
                tok_idx = node.get("token_indices") or []
                tok_idx_remap = []
                for idx in tok_idx:
                    if idx < 0 or idx >= unpadded_len:
                        continue
                    tok_idx_remap.append(idx + pad_len)
                children = node.get("children") or []
                return {
                    "node_type": node.get("node_type", "node"),
                    "token_indices": tok_idx_remap,
                    "children": [remap_node(ch) for ch in children],
                }

            tree = remap_node(parsed["tree_structure"])

            # Update token mask
            upd = parsed.get("update_token_indices") or []
            for idx in upd:
                if 0 <= idx < unpadded_len:
                    update_masks[b, idx + pad_len] = 1.0

            remapped_trees.append(tree)

        # Build chunk memory per height
        chunk_by_h, rb_by_h = self.parse_tree_encoder(token_embeds, remapped_trees)

        # Assign to helpers by matching layer_idx == height
        for layer_idx, helper in self._helpers.items():
            helper.chunk_tensor = chunk_by_h.get(layer_idx)
            helper.chunk_right_bounds = rb_by_h.get(layer_idx)
            helper.token_update_mask = update_masks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        parsed_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):
        self.set_chunk_representations(input_ids, attention_mask, parsed_list)
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
