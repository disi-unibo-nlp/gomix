import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data as GeometricData


class Net(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = SAGEConv(2560, 10000)
        self.batchnorm1 = BatchNorm(10000)
        self.conv2 = SAGEConv(10000, num_classes)

    def forward(self, data: GeometricData):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        return x
