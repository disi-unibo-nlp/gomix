import torch


def load_protein_embedding(embeddings_dir, prot_id):
    d = torch.load(f'{embeddings_dir}/{prot_id}.pt')['mean_representations']
    return d[max(d, key=int)]
