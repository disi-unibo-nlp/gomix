import os
import torch
import pickle


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
ALL_PROTEIN_SEQUENCE_EMBEDDINGS_DIR = os.path.join(THIS_DIR, '../../data/processed/task_datasets/2016/all_protein_sequence_embeddings/esm2_t48_15B_UR50D')
ALL_PROTEIN_TEXT_FEATURES_EMBEDDINGS_DIR = os.path.join(THIS_DIR, '../../data/processed/task_datasets/2016/all_protein_text_features_OpenAI_embeddings')

PROT_SEQUENCE_EMBEDDING_SIZE = 5120  # Number of elements in a single protein embedding vector (`2560` for esm2-3B embeddings, `5120` for esm2-15B embeddings)
PROT_TEXT_EMBEDDING_SIZE = 1536  # Number of elements in a single protein text features embedding vector
WHOLE_PROT_EMBEDDING_SIZE = PROT_SEQUENCE_EMBEDDING_SIZE + PROT_TEXT_EMBEDDING_SIZE


def load_whole_prot_embedding(prot_id) -> torch.Tensor:
    return torch.cat((load_protein_sequence_embedding(prot_id), load_protein_text_embedding(prot_id)))


def load_protein_sequence_embedding(prot_id) -> torch.Tensor:
    d = torch.load(f'{ALL_PROTEIN_SEQUENCE_EMBEDDINGS_DIR}/{prot_id}.pt')['mean_representations']
    d = d[max(d, key=int)]
    assert type(d) == torch.Tensor and d.shape == (PROT_SEQUENCE_EMBEDDING_SIZE,)
    return d


def load_protein_text_embedding(prot_id) -> torch.Tensor:
    pickle_file_path = os.path.join(ALL_PROTEIN_TEXT_FEATURES_EMBEDDINGS_DIR, f'{prot_id}.pickle')
    if not os.path.exists(pickle_file_path):
        return torch.zeros(PROT_TEXT_EMBEDDING_SIZE)

    with open(pickle_file_path, 'rb') as f:
        embedding = pickle.load(f)
    assert len(embedding) == PROT_TEXT_EMBEDDING_SIZE
    return torch.Tensor(embedding)
