**ESM2** is a protein language model by Facebook. It can embed proteins (it first embeds amino acids, then you usually take the average between them and you get the protein embedding).

Facebook provided a script for embedding the proteins from a FASTA file, that I copied into `src/utils/embed_proteins_from_fasta`. Usage is described [here](https://github.com/facebookresearch/esm).

Example: `python src/utils/embed_proteins_from_fasta.py esm2_t33_650M_UR50D data/raw/CAFA3_training_data/uniprot_sprot_exp.fasta data/processed/CAFA3_training_data/protein_embeddings --include mean`

ESM2 model names available: https://huggingface.co/facebook/esm2_t33_650M_UR50D

## Processing annotation file

Use the script `src/utils/process_protein_annotations.py` to process an annotation file. Example usage: `python src/utils/process_protein_annotations.py data/raw/CAFA3_training_data/uniprot_sprot_exp.txt data/processed/CAFA3_training_data/protein_annotations.csv`

## Where do files come from?

`data/processed/CAFA3_training_data/protein_representation/STRING_v11.0_network.json` is the result of running `src/utils/prepare_STRING_network_from_raw_file.py`, which requires as arguments:

- `protein_ids_file_path`: path to a json file containing a list of protein IDs (those that we're interested about, e.g. the prots from the CAFA dataset). This makes the resulting file lighter.
- `network_file_path`: path to the `STRING_vX_network.json` file (it should contain links that involve all the protein IDs you're interested about).
- `mapping_file_path`: uniprot_2_string file path (doesn't need to be that recent).
