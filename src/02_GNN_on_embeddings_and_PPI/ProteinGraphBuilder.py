import json
import os
import numpy as np
import torch
from torch_geometric.data import Data as GeometricData


class ProteinGraphBuilder:
    def __init__(self, ppi_file_path: str):
        # load protein-protein similarity network
        # format: { protein1: { protein2: score2, protein3: score3, ... }, ... }
        with open(ppi_file_path) as fp:
            self.ppi_net = json.load(fp)

        self.prot_ids_list = list(self.ppi_net.keys())
        self.targets = None

    def load_targets(self, prot_annotations_file_path: str):
        # Load protein annotations, i.e. a mapping from protein -> GO term
        # Format: { protein1: [term1, term2], protein2: [term3] }
        with open(prot_annotations_file_path, 'r') as f:
            prot_annotations = json.load(f)

        # First, establish a vocabulary of GO terms.
        go_terms = set()
        for prot_id, ann_go_terms in prot_annotations.items():
            if prot_id in self.prot_ids_list:
                go_terms.update(ann_go_terms)
        go_terms_list = list(go_terms)

        # Then, reuse the vocabulary for setting protein targets (as indexes referring to GO terms in the vocabulary).
        self.targets = []
        for prot_id in self.prot_ids_list:
            target = torch.zeros(len(go_terms_list))

            if prot_id in prot_annotations:
                target[[go_terms_list.index(go_term) for go_term in prot_annotations[prot_id]]] = 1

            self.targets.append(target)

        self.targets = torch.stack(self.targets)
        return {'num_classes': len(go_terms_list)}

    def build(self) -> GeometricData:
        adj_matrix = self._make_adj_matrix()

        adj_matrix[adj_matrix < 0.9] = 0

        edge_index = torch.LongTensor(np.nonzero(adj_matrix))
        # Uncomment if you want to also consider edge weights (i.e. similarity scores). Binary should be enough, though.
        # edge_attr = torch.FloatTensor(adj_matrix[np.nonzero(adj_matrix)])

        x = torch.stack([self._load_protein_embedding(prot_id) for prot_id in self.prot_ids_list])
        assert(x.size(0) == len(self.prot_ids_list))

        if self.targets is None:
            raise 'Targets are missing in graph builder.'

        return GeometricData(
            edge_index=edge_index,
            # edge_attr=edge_attr,
            x=x,
            y=self.targets
        )

    def _make_adj_matrix(self):
        """
        Construct the adjacency matrix
        """
        num_proteins = len(self.prot_ids_list)

        adj_matrix = np.zeros((num_proteins, num_proteins))
        for i, prot1 in enumerate(self.prot_ids_list):
            for j, prot2 in enumerate(self.prot_ids_list):
                if prot1 != prot2 and prot2 in self.ppi_net[prot1]:
                    adj_matrix[i, j] = self.ppi_net[prot1][prot2]

        return adj_matrix

    def _load_protein_embedding(self, prot_id):
        embedding_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../data/processed/CAFA3_training_data/protein_representation/protein_embeddings_esm2_t36_3B_UR50D',
            f'{prot_id}.pt'
        )
        d = torch.load(embedding_file_path)['mean_representations']
        return d[max(d, key=int)]
