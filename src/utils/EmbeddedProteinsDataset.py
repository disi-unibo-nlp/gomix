from torch.utils.data import Dataset
import torch
from src.utils.load_protein_embedding import load_protein_embedding


def _make_go_term_vocabulary(annotations):
    go_terms = set()
    for _, ann_go_terms in annotations.items():
        go_terms.update(ann_go_terms)
    go_terms = list(go_terms)
    return {go_term: i for i, go_term in enumerate(go_terms)}


class EmbeddedProteinsDataset(Dataset):
    def __init__(self, annotations: dict, embeddings_dir):
        self.prot_ids = list(annotations.keys())
        self.annotations = annotations
        self.embeddings_dir = embeddings_dir
        self.go_term_to_index = _make_go_term_vocabulary(annotations)

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_id = self.prot_ids[idx]

        prot_embedding = load_protein_embedding(self.embeddings_dir, prot_id)

        target = torch.zeros(len(self.go_term_to_index))
        for go_term in self.annotations[prot_id]:
            target[self.go_term_to_index[go_term]] = 1

        return prot_embedding, target
