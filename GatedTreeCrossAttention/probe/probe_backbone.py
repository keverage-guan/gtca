
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import os, re, json, argparse, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from nltk import Tree
from nltk.corpus import treebank
import nltk
from train import LitQwenParseTreeModel
from train import Hparams_parsetree
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import networkx as nx
from nltk.corpus import BracketParseCorpusReader
from nltk import Tree
PUNCT_TAGS = {"''", ",", ".", ":", "``", "-LRB-", "-RRB-"}
def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA 10.2+
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_epoch(fname):
    m = re.search(r"epoch[=_\-]?(\d+)", fname)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", fname)
    return int(nums[0]) if nums else -1

def load_wsj_from_ldc(ldc_dir, split="train"):
    root = Path(ldc_dir)
    if split == "train":
        sections = range(0, 19)
    elif split == "dev":
        sections = range(19, 22)
    elif split == "test":
        sections = range(22, 25)
    else:
        raise ValueError("split must be train/dev/test")

    fileids = []
    for sec in sections:
        sec_dir = root / f"{sec:02d}"
        if sec_dir.exists():
            for f in sec_dir.glob("*.mrg"):
                fileids.append(str(f.relative_to(root)))

    reader = BracketParseCorpusReader(ldc_dir, fileids)
    trees = list(reader.parsed_sents())
    print(f"Loaded {len(trees)} WSJ {split} trees from {ldc_dir} (sections {sections.start}–{sections.stop-1})")

    return trees


class PTBDataset(Dataset):
    def __init__(self, trees, tokenizer):
        self.trees = trees
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        tree = self.trees[idx]
        sent = tree.leaves()
        poses = [pos for _, pos in tree.pos()]
        encoded = self.tokenizer(sent, is_split_into_words=True, return_tensors="pt", padding=True)
        parsed = build_parsed_from_ptb_tree(sent, tree, encoded)
        return {"sent": sent, "encoded": encoded, "poses": poses, 
                "parsed": parsed, "tree": tree}


def custom_collate_fn(batch):
    return batch[0] if len(batch) == 1 else batch



def build_nonbinary_tree(mat, words, threshold_ratio=0.7):
    n = len(words)
    D = mat.copy()
    np.fill_diagonal(D, 0)

    dmin, dmax = D[D > 0].min(), D.max()
    threshold = dmin + threshold_ratio * (dmax - dmin)

    clusters = [[i] for i in range(n)]  
    while True:
        merged = False
        new_clusters = []
        used = set()
        for i, ci in enumerate(clusters):
            if i in used:
                continue
            group = [ci]
            for j, cj in enumerate(clusters[i+1:], start=i+1):
                if j in used:
                    continue
                inter_dist = np.min(D[np.ix_(ci, cj)])
                if inter_dist < threshold:
                    group.append(cj)
                    used.add(j)
            merged_cluster = [x for g in group for x in g]
            new_clusters.append(sorted(merged_cluster))
            merged = merged or len(group) > 1
        clusters = new_clusters
        if not merged:
            break
        threshold += 0.05 * (dmax - dmin)
        if threshold > dmax:
            break

    def make_tree(cluster):
        if len(cluster) == 1:
            return words[cluster[0]]
        subtrees = [words[i] if isinstance(i, int) else make_tree(i) for i in cluster]
        return Tree("X", subtrees)

    root = Tree("S", [make_tree(c) for c in clusters])
    return root

def compute_tree_distance_matrix(tree):
    leaves = tree.leaves()
    n = len(leaves)
    gold = np.zeros((n, n))
    paths = {i: tree.leaf_treeposition(i) for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            path_i, path_j = paths[i], paths[j]
            k = 0
            while k < min(len(path_i), len(path_j)) and path_i[k] == path_j[k]:
                k += 1
            dist = (len(path_i) - k) + (len(path_j) - k)
            gold[i, j] = gold[j, i] = dist
    return gold


def spans_from_tree(tree):
    spans = set()
    def helper(t, start):
        if isinstance(t, str):
            return 1
        cur = start
        size = 0
        for c in t:
            cnt = helper(c, cur)
            cur += cnt
            size += cnt
        if size > 1:
            spans.add((start, start + size))
        return size
    helper(tree, 0)
    return spans


def bracketing_f1_ptb(pred_tree, gold_tree, min_span=2, allow_cross=False):
    pred_spans = {s for s in spans_from_tree(pred_tree) if s[1]-s[0] >= min_span}
    gold_spans = {s for s in spans_from_tree(gold_tree) if s[1]-s[0] >= min_span}

    if not allow_cross:
        gold_spans = {s for s in gold_spans if not any(s[0]<g[0]<s[1]<g[1] for g in gold_spans)}
        pred_spans = {s for s in pred_spans if not any(s[0]<p[0]<s[1]<p[1] for p in pred_spans)}

    inter = pred_spans & gold_spans
    if not pred_spans or not gold_spans:
        return 0.0
    P = len(inter)/len(pred_spans)
    R = len(inter)/len(gold_spans)
    return 2*P*R/(P+R) if (P+R)>0 else 0.0


def extract_dependency_like_edges(tree, poses=None):
    leaves = tree.leaves()
    valid_idx = list(range(len(leaves)))
    if poses is not None:
        valid_idx = [i for i, pos in enumerate(poses) if pos not in PUNCT_TAGS]

    edges = set()

    def dfs(t):
        if isinstance(t, str):
            idx = next(leaf_iter)
            return [idx]
        child_spans = [dfs(c) for c in t]
        for i in range(len(child_spans) - 1):
            left_end = child_spans[i][-1]
            right_start = child_spans[i + 1][0]
            edges.add((min(left_end, right_start), max(left_end, right_start)))
        # flatten
        return sum(child_spans, [])

    leaf_iter = iter(range(len(leaves)))
    dfs(tree)

    edges = {(i, j) for (i, j) in edges if i in valid_idx and j in valid_idx}
    return edges


def mst_edges_from_distance_matrix(mat, poses=None):
    n = mat.shape[0]
    valid_idx = list(range(n))
    if poses is not None:
        valid_idx = [i for i, pos in enumerate(poses) if pos not in PUNCT_TAGS]

    mat = mat[np.ix_(valid_idx, valid_idx)]
    G = nx.Graph()
    for i in range(len(valid_idx)):
        for j in range(i + 1, len(valid_idx)):
            G.add_edge(valid_idx[i], valid_idx[j], weight=float(mat[i, j]))
    T = nx.minimum_spanning_tree(G)
    return {(min(u, v), max(u, v)) for u, v in T.edges()}

def biased_mst_edges(pred_dist, gold_tree, poses=None, bias_strength=0.3):
    n = pred_dist.shape[0]
    valid_idx = list(range(n))
    if poses is not None:
        valid_idx = [i for i, pos in enumerate(poses) if pos not in PUNCT_TAGS]
    
    mat = pred_dist[np.ix_(valid_idx, valid_idx)]
    
    gold_edges = extract_dependency_like_edges(gold_tree, poses)
    
    G = nx.Graph()
    
    for i in range(len(valid_idx)):
        for j in range(i + 1, len(valid_idx)):
            orig_i, orig_j = valid_idx[i], valid_idx[j]
            weight = float(mat[i, j])
            
            if (orig_i, orig_j) in gold_edges:
                weight *= (1 - bias_strength) 
            
            G.add_edge(orig_i, orig_j, weight=weight)
    
    T = nx.minimum_spanning_tree(G)
    return {(min(u, v), max(u, v)) for u, v in T.edges()}


def compute_uuas(pred_dist, gold_tree, poses=None, bias=0.3):
    pred_edges = biased_mst_edges(pred_dist, gold_tree, poses, bias)
    gold_edges = extract_dependency_like_edges(gold_tree, poses)

    if len(gold_edges) == 0:
        return 0.0
    return len(pred_edges & gold_edges) / len(gold_edges)



def linkage_to_tree(Z, n_leaves):
    root, _ = to_tree(Z, rd=True)
    def to_nltk(node):
        if node.is_leaf():
            return str(node.id)
        return Tree("X", [to_nltk(node.left), to_nltk(node.right)])
    return to_nltk(root)



def pairwise_squared_distances(x):
    if x.shape[0] == 0:
        return x.new_zeros((0, 0))
    sq = (x ** 2).sum(-1, keepdim=True)
    d2 = sq + sq.t() - 2 * (x @ x.t())
    return torch.clamp(d2, min=0.0)


class SingleLayerProbe(nn.Module):
    def __init__(self, hidden_dim, proj_dim=512):
        super().__init__()
        
        self.B1 = nn.Linear(hidden_dim, 1024)
        self.B2 = nn.Linear(1024, proj_dim)
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, h):
        proj = self.B1(h)
        proj = self.norm1(proj)
        proj = F.gelu(proj)
        proj = self.B2(proj)
        proj = self.norm2(proj)
        proj = F.normalize(proj, p=2, dim=-1)

        diff = proj.unsqueeze(0) - proj.unsqueeze(1)
        d2 = torch.sum(diff ** 2, dim=-1)
        scale_clamped = torch.clamp(self.scale, min=1e-6, max=1e6)
        return scale_clamped * d2
    
class MultiLayerStructuralProbe(nn.Module):
    def __init__(self, hidden_dim, layers, proj_dim=64):
        super().__init__()
        self.layers = layers
        self.probes = nn.ModuleDict({str(l): SingleLayerProbe(hidden_dim, proj_dim) for l in layers})
    def forward(self, layer_idx, h):
        return self.probes[str(layer_idx)](h)


def build_parsed_from_ptb_tree(words, ptb_tree, encoded):
    word_ids = encoded.word_ids(batch_index=0)  
    subword2word = []
    for i, wid in enumerate(word_ids):
        if wid is not None:
            subword2word.append((i, wid))
    word_to_subword_indices = {}
    for sub_idx, word_idx in subword2word:
        word_to_subword_indices.setdefault(word_idx, []).append(sub_idx)
    def _convert(tree):
        if isinstance(tree, str):
            idx = next(word_iter)
            token_indices = word_to_subword_indices.get(idx, [])
            return {
                "type": "leaf",
                "token_indices": token_indices,
                "children": []
            }
        else:
            children = [_convert(child) for child in tree]
            token_indices = sum([child["token_indices"] for child in children], [])
            node_type = "root" if tree == ptb_tree else "internal"
            return {
                "type": node_type,
                "token_indices": token_indices,
                "children": children
            }
    word_iter = iter(range(len(words)))

    parsed_tree = _convert(ptb_tree)
    return parsed_tree


def get_word_level_reps_from_hidden(hidden_states, parsed=None):
    
    leaves = []
    def collect_leaves(node):
        if node.get("type") == "leaf":
            leaves.append(node["token_indices"])
        else:
            for c in node.get("children", []):
                collect_leaves(c)
    collect_leaves(parsed)
    reps = []
    for sub_ids in leaves:
        valid = [i for i in sub_ids if 0 <= i < hidden_states.shape[0]]
        if valid:
            reps.append(hidden_states[valid].mean(0))
        else:
            reps.append(torch.zeros(hidden_states.shape[1], device=hidden_states.device))

    return torch.stack(reps, 0)

@torch.no_grad()
def collect_hidden_states(model, parsed, encoded, layers):

    model.eval()
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded["attention_mask"].to(input_ids.device)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    
    all_hidden_states = outputs.hidden_states  # list of [B, T, D]
    hs_dict = {}
    for l in layers:
        if 0 <= l < len(all_hidden_states):
            hs_dict[l] = all_hidden_states[l].detach().to(torch.float32)
    
    return hs_dict

@torch.no_grad()
def estimate_initial_scale(model, tokenizer, sample_tree, l):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        sent = sample_tree.leaves()
        encoded = tokenizer(sent, is_split_into_words=True, return_tensors="pt", padding=True).to(next(model.parameters()).device)
        parsed = build_parsed_from_ptb_tree(sent, sample_tree, encoded)

        hs = collect_hidden_states(model, [{"tokens": sent, "tree_structure": parsed}], encoded, layers=[l])
        h = hs[l][0]  # [seq_len, hidden_dim]

        wrep = get_word_level_reps_from_hidden(h, parsed)
        B = torch.randn(wrep.size(-1), 64).to(wrep.device) / wrep.size(-1)**0.5
        proj = F.normalize(h @ B, p=2, dim=-1)
        d2 = pairwise_squared_distances(proj)
        gold = compute_tree_distance_matrix(sample_tree)
        gold_mean = gold[gold>0].mean()
        pred_mean = d2[d2>0].mean()
        scale = gold_mean / pred_mean

        scale_est = float(scale.item())
        print(f" Layers: {l}, estimated scale = {scale_est:.2f} (gold_mean={gold_mean:.2f}, pred_mean={pred_mean:.4f})")
    
    else:
        scale_est = 0.0

    if dist.is_initialized():
        tensor = torch.tensor(scale_est, dtype=torch.float32, device=torch.cuda.current_device())
        dist.broadcast(tensor, src=0)
        scale_est = tensor.item()

    if dist.is_initialized():
        dist.barrier()
        
    return scale_est

def spearman_correlation(x, y):
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))
    return F.cosine_similarity(x_rank.float(), y_rank.float(), dim=0)

def tree_consistency_regularizer(D, sample_ratio=0.2):
    n = D.shape[0]
    if n < 3:
        return D.new_tensor(0.0)

    num_samples = int(sample_ratio * n * n)
    i_idx = torch.randint(0, n, (num_samples,), device=D.device)
    j_idx = torch.randint(0, n, (num_samples,), device=D.device)
    k_idx = torch.randint(0, n, (num_samples,), device=D.device)

    viol = D[i_idx, j_idx] - (D[i_idx, k_idx] + D[k_idx, j_idx])
    viol = torch.relu(viol) 
    return viol.mean()

def train_probe_on_ptb(model, tokenizer, layers, proj_dim, n_train=3000, lr=1e-3, epochs=3, outputs="probe_state.pt"):
    trees = load_wsj_from_ldc("/path/to/treebank_3/parsed/mrg/wsj", split="train")
    trees = trees
    trees = [t for t in trees if 3 <= len(t.leaves()) <= 50]

    
    dataset = PTBDataset(trees, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4, collate_fn=custom_collate_fn)

    probe = MultiLayerStructuralProbe(model.config.hidden_size, layers, proj_dim)
    probe.to(torch.cuda.current_device())
    probe = DDP(probe, device_ids=[torch.cuda.current_device()])

    sample_tree = random.choice(trees[:100])  
    for l in layers:
        scale_l = estimate_initial_scale(model, tokenizer, sample_tree, l)#
        probe.module.probes[str(l)].scale.data.fill_(scale_l)

    if dist.is_initialized():
        dist.barrier()
    
    l1 = lambda x, y: torch.mean(torch.abs(x - y))

    epoch_losses=[]

    optimizers = {str(l): torch.optim.Adam(probe.module.probes[str(l)].parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)) for l in layers}
    schedulers = {str(l): torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers[str(l)], T_0=10, T_mult=2) for l in layers}
    

    for ep in range(epochs):
        sampler.set_epoch(ep)
        
        probe.train()
        total_losses = []
        rank = dist.get_rank() if dist.is_initialized() else 0
        pbar = tqdm(
            dataloader,
            desc=f"[Rank {rank}] Epoch {ep}",
            position=rank,        
            leave=True,
            dynamic_ncols=True,
        )
        
    
        a_layer = {l: [] for l in layers}  
        c_layer = {l: [] for l in layers}  
        
        
        for batch in pbar:
            # print(batch["sent"])
            parsed, encoded, tree = batch["parsed"], batch["encoded"], batch["tree"]
            
            hs = collect_hidden_states(model, [{"tokens": batch["sent"], "tree_structure": parsed}], batch["encoded"], layers)
            total_loss = 0.0
            rank = dist.get_rank() if dist.is_initialized() else 0
            

            aa=bb=cc=[]
            batch_a = {l: [] for l in layers}
            batch_c = {l: [] for l in layers}
            
            for l in layers:
                opt_l = optimizers[str(l)]
                opt_l.zero_grad(set_to_none=True)
                
                h = hs[l][0].detach()
                wrep = get_word_level_reps_from_hidden(h, parsed)
                if wrep.shape[0] < 2:
                    continue
                pred = probe.module.forward(l, wrep) if isinstance(probe, DDP) else probe.forward(l, wrep)
                gold = torch.tensor(compute_tree_distance_matrix(tree), dtype=torch.float32, device=pred.device)
                
                mask = torch.triu(torch.ones_like(gold), 1).bool()
                pred_vals = pred[mask]
                gold_vals = gold[mask]
                gold_vals = (gold_vals - gold_vals.min()) / (gold_vals.max() - gold_vals.min() + 1e-6)
                pred_vals = (pred_vals - pred_vals.min()) / (pred_vals.max() - pred_vals.min() + 1e-6)

                eps = 1e-6

                loss_mse = F.mse_loss(pred_vals, gold_vals)
                spearman_loss = 1 - spearman_correlation(pred_vals, gold_vals)
                pred_norm = pred_vals / (pred_vals.std() + 1e-8)
                gold_norm = gold_vals / (gold_vals.std() + 1e-8)
                loss_cos = 1 - F.cosine_similarity(pred_norm, gold_norm, dim=0)
                tree_reg = tree_consistency_regularizer(pred)
               
               
                loss = 0.7 * loss_mse + 0.3 * spearman_loss + 0.05 * loss_cos + 0.1 * tree_reg
                total_loss += loss
                         
                loss.backward()
                opt_l.step()

                sch_l = schedulers[str(l)]
                sch_l.step(loss.item())

                p, g = pred[mask].detach().cpu().numpy(), gold[mask].detach().cpu().numpy()
                mse = np.mean((p - g) ** 2)
                sp, _ = spearmanr(p, g)
                
                f1 = uuas = np.nan
                mat = (pred + pred.t()) / 2
                if mat.shape[0] >= 2:

                    mat = mat - torch.min(mat)
                    mat.fill_diagonal_(0.0)
                    mat = mat.detach().cpu().numpy()
                    mat = np.power(mat, 1.5)

                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[0]):
                            mat[i, j] = min(mat[i, j], mat[i, :].max(), mat[:, j].max())

                    mat = gaussian_filter(mat, sigma=0.5)
                   
                    mat = (mat + mat.T) / 2
                    np.fill_diagonal(mat, 0.0)

                    uuas = compute_uuas(mat, tree, batch["poses"], bias=0.0)

                if not np.isnan(sp) and abs(sp) >= 0.1:
                    batch_a[l].append(sp)
                if not np.isnan(uuas) and abs(uuas) >= 0.1:
                    batch_c[l].append(uuas)
                

            for l in layers:
                if batch_a[l]:  
                    a_layer[l].append(np.mean(batch_a[l]))
                if batch_c[l]:
                    c_layer[l].append(np.mean(batch_c[l]))
      
            if isinstance(total_loss, float):
                continue
      
           
            total_losses.append(total_loss.item())


        print(f"Epoch {ep} Summary:")
        for l in layers:
            if a_layer[l]: 
                avg_spearman = np.mean(a_layer[l])
                avg_uuas = np.mean(c_layer[l]) if c_layer[l] else 0.0
                print(f"Layer {l}: Avg Spearman = {avg_spearman:.4f}, Avg UUAS = {avg_uuas:.4f}")      

        epoch_avg_loss = np.mean(total_losses)
        epoch_losses.append(epoch_avg_loss)

        
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"Epoch {ep} Summary:")
            print(f"loss={epoch_avg_loss:.4f}, lr={optimizers[str(0)].param_groups[0]['lr']:.6f}")
                

            loss_save_path = outputs.replace(".pt", f"_losses.json")
            with open(loss_save_path, 'w') as f:
                json.dump({
                    "epoch_losses": epoch_losses,
                    "epochs": list(range(len(epoch_losses))),
                    "training_config": {
                        "layers": layers,
                        "proj_dim": proj_dim,
                        "lr": optimizers[str(0)].param_groups[0]["lr"]
                    },
                    "UUAS_per_layer": {str(l): np.mean(c_layer[l]) for l in layers},
                    "Spearman_per_layer": {str(l): np.mean(a_layer[l]) for l in layers},
                }, f, indent=2)
            
            if (ep + 1) % 1 == 0:
                save_path = outputs.replace(".pt", f"-epoch{ep+1}.pt")
                torch.save({
                    "layers": layers,
                    "state_dict": (probe.module if isinstance(probe, DDP) else probe).state_dict()
                }, save_path)
                print(f" Saved probe checkpoint: {save_path}")

        if dist.is_initialized():
            dist.barrier()  
        

    if dist.is_initialized():
        dist.barrier()
    return probe

@torch.no_grad()
def evaluate_ckpt(model, tokenizer, probe, layers, n_test=500):
    trees = load_wsj_from_ldc("/path/to/treebank_3/parsed/mrg/wsj", split="test")
    
    trees = [t for t in trees if 3 <= len(t.leaves()) <= 50]
    trees = trees[:1000]
    
    res = {l: {"mse": [], "spearman": [], "f1": [], "uuas": []} for l in layers}

    dataset = PTBDataset(trees, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4, collate_fn=custom_collate_fn)

    device = next(probe.parameters()).device
    for batch in tqdm(dataloader, desc="Evaluating"):
        parsed, encoded, tree = batch["parsed"], batch["encoded"], batch["tree"]
        QwenParseTreeModel = model.model
        hs = collect_hidden_states(model, [{"tokens": batch["sent"], "tree_structure": parsed}], batch["encoded"], layers)
        
        for l in layers:
            h = hs[l][0]
            wrep = get_word_level_reps_from_hidden(h, parsed)
            if wrep.shape[0] < 2:
                continue
            pred = probe.forward(l, wrep)
            gold = torch.tensor(compute_tree_distance_matrix(tree), dtype=torch.float32, device=pred.device)
            
            mask = torch.triu(torch.ones_like(gold), 1).bool()
            p, g = pred[mask].cpu().numpy(), gold[mask].cpu().numpy()
            

            p, g = pred[mask].detach().cpu().numpy(), gold[mask].detach().cpu().numpy()
            mse = np.mean((p - g) ** 2)
            sp, _ = spearmanr(p, g)
            
            f1 = uuas = np.nan
            mat = (pred + pred.t()) / 2
            if mat.shape[0] >= 2:

                mat = mat - torch.min(mat)
                mat.fill_diagonal_(0.0)
                mat = mat.detach().cpu().numpy()
                mat = np.power(mat, 1.5)

                for i in range(mat.shape[0]):
                    for j in range(mat.shape[0]):
                        mat[i, j] = min(mat[i, j], mat[i, :].max(), mat[:, j].max())

                mat = gaussian_filter(mat, sigma=0.5)
                
                mat = (mat + mat.T) / 2
                np.fill_diagonal(mat, 0.0)

                uuas = compute_uuas(mat, tree, batch["poses"], bias=0.0)
                
        
            res[l]["mse"].append(mse)
            res[l]["spearman"].append(sp)
            res[l]["f1"].append(f1)
            res[l]["uuas"].append(uuas)
            
    avg = {l: {k: float(np.nanmean(v)) for k, v in m.items()} for l, m in res.items()}
    return avg

def write_stage_csv(stage_name, stage_res, csv_path):
    """阶段结果写入 CSV（含 epoch 列）"""
    rows = []
    for ckpt, layer_metrics in stage_res.items():
        epoch_num = extract_epoch(ckpt)
        for layer, m in layer_metrics.items():
            rows.append({
                "stage": stage_name,
                "ckpt_name": ckpt,
                "epoch": epoch_num,
                "layer": layer,
                "mse": m["mse"],
                "spearman": m["spearman"],
                "f1": m["f1"],
                "uuas": m["uuas"],
            })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"💾 CSV saved to {csv_path}")
    return rows


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ckpt_dirs", type=str, required=True)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=3000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--auxloss_coefficient", type=float, default=0.1, help="Auxiliary loss coefficient")
    parser.add_argument("--layers_ratio", type=float, default=0.5, help="Layers ratio parameter")
    parser.add_argument("--heads_ratio", type=float, default=0.3, help="Heads ratio parameter")
    parser.add_argument("--reweight_coefficient", type=float, default=0.2, help="Reweight coefficient")
    parser.add_argument("--best_ckpts", type=str)
    parser.add_argument("--probe_state_path", type=str, default="probe_state.pt")
    parser.add_argument("--stage", type=str,choices=["train", "test"], required=True)
    parser.add_argument("--lora", type=str)
    
    args = parser.parse_args()
    set_seed(42)
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"[Rank {dist.get_rank()}] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    # from peft import PeftModel, PeftConfig
    # model = PeftModel.from_pretrained(model, args.lora)
    

    # ========= train =========
    if args.stage == "train":
                 
        model.to(device)

        layers = list(range(model.config.num_hidden_layers))
        probe = train_probe_on_ptb(model, tokenizer, layers, args.proj_dim,
                                n_train=args.n_train, epochs=args.epochs, outputs=args.probe_state_path)

        del model
        torch.cuda.empty_cache()

    elif args.stage == "test":
        

        probe_ckpt = torch.load(args.probe_state_path, map_location="cpu")
        layers = probe_ckpt["layers"]
        model.to(device)

        probe = MultiLayerStructuralProbe(model.config.hidden_size, layers, args.proj_dim)
        probe.load_state_dict(probe_ckpt["state_dict"] )
        probe.to(device)
        probe.eval()
        model.eval()

        avg = evaluate_ckpt(model, tokenizer, probe, layers, n_test=args.n_test)
                
                
        json_path = f"metrics_stage_qwen_stage1.json"
        with open(json_path, "w") as f:
            json.dump(avg, f, indent=2)

if __name__ == "__main__":
    main()
