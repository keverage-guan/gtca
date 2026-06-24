# model/dep_labels.py
"""Shared dependency-relation vocabulary for the dependency-tree GTCA variant."""

# spaCy English `dep_` label set. Index 0 is reserved for <none>/<unk>
DEP_LABELS = [
    "<none>", "ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod",
    "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound",
    "conj", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl",
    "intj", "mark", "meta", "neg", "nmod", "npadvmod", "nsubj", "nsubjpass",
    "nummod", "oprd", "parataxis", "pcomp", "pobj", "poss", "preconj",
    "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp",
]
_DEP_LABEL_TO_ID = {lab: i for i, lab in enumerate(DEP_LABELS)}

# 0 = none/pad, 1 = root, 2 = head-to-left, 3 = head-to-right
DEP_DIRS = ["<none>", "root", "head_left", "head_right"]
NUM_DEP_LABELS = len(DEP_LABELS)
NUM_DEP_DIRS = len(DEP_DIRS)

def dep_label_to_id(label: str) -> int:
    return _DEP_LABEL_TO_ID.get(label, 0)

def dep_dir_to_id(direction: str) -> int:
    return {"root": 1, "head_left": 2, "head_right": 3}.get(direction, 0)