## ESM2 + PPI net -> GCN -> dense

The idea is using PPI network information (a bit like HPODNets did).

Differently from HPODNets, we're going to initialize the graph nodes with ESM2 protein embeddings, which are then processed by a GNN.
