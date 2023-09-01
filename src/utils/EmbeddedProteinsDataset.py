from torch.utils.data import Dataset
import torch
from typing import Callable


def _make_go_term_vocabulary(annotations):
    go_terms = set()
    for _, ann_go_terms in annotations.items():
        go_terms.update(ann_go_terms)
    go_terms = list(go_terms)
    return {go_term: i for i, go_term in enumerate(go_terms)}


class EmbeddedProteinsDataset(Dataset):
    def __init__(self, annotations: dict, load_prot_embedding: Callable[[str], torch.Tensor]):
        self.prot_ids = list(annotations.keys())
        self.annotations = annotations
        self.go_term_to_index = _make_go_term_vocabulary(annotations)
        self._load_prot_embedding = load_prot_embedding

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_id = self.prot_ids[idx]

        prot_embedding = self._load_prot_embedding(prot_id)

        target = torch.zeros(len(self.go_term_to_index))
        for go_term in self.annotations[prot_id]:
            target[self.go_term_to_index[go_term]] = 1

        return prot_embedding, target
