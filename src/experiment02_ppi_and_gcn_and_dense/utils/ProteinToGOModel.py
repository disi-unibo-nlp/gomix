#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import Module, Parameter, Dropout
import torch.nn.functional as F
from torch_geometric.nn import dense, norm
from torch_geometric.nn.inits import glorot
from math import log

torch.manual_seed(31)


class ProteinToGOModel(nn.Module):
    def __init__(self, n_protein, n_term, dim=500, alpha=0.5, theta=0.5, dropout=0.5):
        super(ProteinToGOModel, self).__init__()

        self.n_protein = n_protein
        self.n_term = n_term

        self.bn_in = norm.BatchNorm(n_protein)
        self.linear_in = nn.Linear(n_protein, dim, bias=True)
        self.gcn1 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=1, dropout=dropout)
        self.gcn2 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=2, dropout=dropout)
        self.gcn3 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=3, dropout=dropout)
        self.gcn4 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=4, dropout=dropout)
        self.gcn5 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=5, dropout=dropout)
        self.gcn6 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=6, dropout=dropout)
        self.gcn7 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=7, dropout=dropout)
        self.gcn8 = DeepGCNLayer(dim, alpha=alpha, theta=theta, layer=8, dropout=dropout)
        self.linear_out = nn.Linear(dim, n_term)

    def forward(self, x, net):
        x = torch.squeeze(self.bn_in(x))
        x = F.leaky_relu(self.linear_in(x))
        hidden = self.gcn1(x, x, net)
        hidden = self.gcn2(hidden, x, net)
        hidden = self.gcn3(hidden, x, net)
        hidden = self.gcn4(hidden, x, net)
        hidden = self.gcn5(hidden, x, net)
        hidden = self.gcn6(hidden, x, net)
        hidden = self.gcn7(hidden, x, net)
        hidden = self.gcn8(hidden, x, net)
        pred = self.linear_out(hidden)
        return pred


class GCNII(Module):
    """The graph convolutional operator with initial residual connections and
       identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
       Networks" <https://arxiv.org/abs/2007.02133>`_ paper"""
    def __init__(self, channels, alpha, theta=None, layer=None,
                 shared_weights=True, add_self_loops=True, normalize=True):
        super(GCNII, self).__init__()

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)

    def forward(self, x, x_0, adj):
        N, _ = adj.size()

        if self.add_self_loops:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[idx, idx] = 1

        if self.normalize:
            deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        x = torch.matmul(adj, x)
        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out


class DeepGCNLayer(Module):
    """
    inspired by “DeepGCNs: Can GCNs Go as Deep as CNNs?”
    url: https://arxiv.org/abs/1904.03751
    GraphConv -> Normalization -> Activation -> Dropout
    """
    def __init__(self, channels, alpha=0.5, theta=0.5, layer=None, dropout=0.5):
        super(DeepGCNLayer, self).__init__()

        self.gcn = GCNII(channels, alpha=alpha, theta=theta, layer=layer)
        self.bn = norm.BatchNorm(channels)
        self.dropout = Dropout(p=dropout)

    def forward(self, x, x_0, adj):
        hidden = self.gcn(x, x_0, adj)
        hidden = torch.squeeze(self.bn(hidden))
        hidden = F.leaky_relu(hidden)
        hidden = self.dropout(hidden)
        return hidden
