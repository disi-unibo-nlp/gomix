from torch.utils.data import Dataset
import torch
from src.utils.load_protein_embedding import load_protein_embedding


class EmbeddedProteinsDataset(Dataset):
    def __init__(self, annotations: dict, embeddings_dir, go_term_to_index: dict):
        self.prot_ids = list(annotations.keys())
        self.annotations = annotations
        self.embeddings_dir = embeddings_dir
        self.go_term_to_index = go_term_to_index

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_id = self.prot_ids[idx]

        prot_embedding = load_protein_embedding(self.embeddings_dir, prot_id)

        target = torch.zeros(len(self.go_term_to_index))
        for go_term in self.annotations[prot_id]:
            target[self.go_term_to_index[go_term]] = 1

        return prot_embedding, target
