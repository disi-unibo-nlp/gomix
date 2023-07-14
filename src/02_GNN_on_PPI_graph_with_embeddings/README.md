# ESM2 + PPI net -> GCN

The idea is using PPI network information (a bit like HPODNets did).

Differently from HPODNets, we're going to initialize the graph nodes with ESM2 protein embeddings, which are then processed by a GNN.

## Training results

_Write the results here._

## Ways to improve

- Use [SIGN](https://arxiv.org/pdf/2004.11198.pdf) or [GraphSAINT](https://arxiv.org/abs/1907.04931) instead of the current GCN.
- Combine the current solution with [Proteinfer](https://google-research.github.io/proteinfer/) using ensemble learning such as stacking or Mixture of Experts.
- Include other PPI networks as input.
- Add to the input also the old protein-GO term links (with a specific edge type).