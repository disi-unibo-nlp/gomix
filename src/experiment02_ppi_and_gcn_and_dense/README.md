This experiment attempts to replicate the approach used by the HPODNets paper, i.e. using info from Protein-Protein Interaction networks (and nothing else) to predict the ontology terms for a protein.

In our case, we just use the STRING network, so we changed the model a bit.

Unfortunately, with the HPODNets approach, you have to do full-batch training because the model needs the whole PPI graph to make predictions.

### Current status

Note: Training is slow because of the full-batch training requirement, so I can't run it on Mac's MPS device, but I need CUDA.

Note: Currently, the STRING network file only contains data about around 13k of the proteins that are in the CAFA3 training set. That's because it's just a subset. I have to download the full STRING network (which is heavier). I can then remove the proteins that aren't in CAFA3 to reduce the file size.
