import torch.nn as nn
import torch.nn.functional as F


class ProteinToGOModel(nn.Module):
    def __init__(self, protein_embedding_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(protein_embedding_size, 14000),
            nn.BatchNorm1d(14000),
            nn.ReLU(),

            nn.Linear(14000, 10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),

            nn.Linear(10000, 8000),
            nn.BatchNorm1d(8000),
            nn.ReLU(),

            nn.Linear(8000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),

            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),

            nn.Linear(4000, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        return F.sigmoid(self.forward(x))
