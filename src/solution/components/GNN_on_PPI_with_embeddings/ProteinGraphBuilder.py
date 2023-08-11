import json
import os
import numpy as np
import torch
from torch_geometric.data import Data as GeometricData
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.utils.load_protein_embedding import load_protein_embedding

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
ALL_PROTEIN_EMBEDDINGS_DIR = os.path.join(THIS_DIR, '../../../../data/processed/task_datasets/2016/all_protein_embeddings/esm2_t48_15B_UR50D')


class ProteinGraphBuilder:
    def __init__(self, ppi_file_path: str):
        # load protein-protein similarity network
        # format: { protein1: { protein2: score2, protein3: score3, ... }, ... }
        with open(ppi_file_path) as fp:
            self._ppi_net = json.load(fp)

        self._targets = None

        self.prot_id_to_node_idx = {prot_id: i for i, prot_id in enumerate(self._ppi_net.keys())}
        self.go_term_to_class_idx = None

    def set_targets(self, prot_annotations_file_path: str) -> None:
        # Load protein annotations, i.e. a mapping from protein -> GO term
        # Format: { protein1: [term1, term2], protein2: [term3] }
        with open(prot_annotations_file_path, 'r') as f:
            prot_annotations = json.load(f)

        # First, establish a vocabulary of GO terms.
        go_terms = set()
        for prot_id, ann_go_terms in prot_annotations.items():
            if prot_id in self.prot_id_to_node_idx:
                go_terms |= set(ann_go_terms)
        self.go_term_to_class_idx = {go_term: i for i, go_term in enumerate(go_terms)}

        # Then, reuse the vocabulary for setting protein targets (as indexes of GO terms in the vocabulary).
        self._targets = []
        for prot_id in self._get_sorted_prot_ids():
            target = torch.zeros(len(self.go_term_to_class_idx))

            if prot_id in prot_annotations:
                target[[self.go_term_to_class_idx[go_term] for go_term in prot_annotations[prot_id]]] = 1

            self._targets.append(target)

        self._targets = torch.stack(self._targets)

    def build(self) -> GeometricData:
        adj_matrix = self._make_adj_matrix()

        adj_matrix[adj_matrix < 0.86] = 0

        edge_index = torch.LongTensor(np.nonzero(adj_matrix))
        # Uncomment if you want to also consider edge weights (i.e. similarity scores). Binary should be enough, though.
        # edge_attr = torch.FloatTensor(adj_matrix[np.nonzero(adj_matrix)])

        prot_embeddings = [load_protein_embedding(ALL_PROTEIN_EMBEDDINGS_DIR, prot_id) for prot_id in self._get_sorted_prot_ids()]
        x = torch.stack(prot_embeddings)
        assert(x.size(0) == len(self.prot_id_to_node_idx))

        return GeometricData(
            edge_index=edge_index,
            # edge_attr=edge_attr,
            x=x,
            y=self._targets
        )

    def _get_sorted_prot_ids(self):
        return [k for k, _ in sorted(self.prot_id_to_node_idx.items(), key=lambda item: item[1])]

    def _make_adj_matrix(self):
        """
        Construct the adjacency matrix
        """
        sorted_prot_ids = self._get_sorted_prot_ids()
        num_proteins = len(sorted_prot_ids)

        adj_matrix = np.zeros((num_proteins, num_proteins))
        for i, prot1 in enumerate(sorted_prot_ids):
            for j, prot2 in enumerate(sorted_prot_ids):
                if prot1 != prot2 and prot2 in self._ppi_net[prot1]:
                    adj_matrix[i, j] = self._ppi_net[prot1][prot2]

        return adj_matrix
