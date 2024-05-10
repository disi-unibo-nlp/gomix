# Predicting Protein Functions with Ensemble Deep Learning and Protein Language Models
We present _GOMix_, an ensemble learning method for predicting the functions of newly discovered proteins, packaged within an easy-to-use web application.
By combining established algorithms and cutting-edge embedding models, _GOMix_ achieves competitive or state-of-the-art performance in the [CAFA-3 challenge](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1835-8).
Unlike existing solutions, \textsc{GOMix} is entirely open-source, modular, and computationally low-resource.
In this research work, we tackle the problem of predicting the GO terms associated to a protein, based on the protein's amino-acidic sequence and the way it interacts with other proteins as specified by PPI networks (Protein-Protein Interaction networks), like STRING.

## Dataset

- `data/raw/task_datasets/CAFA3` comes from https://github.com/bio-ontology-research-group/deepgoplus
- `data/raw/Uniprot/uniprot_swiss_entries.dat` comes from https://www.uniprot.org/help/downloads
- `data/raw/InterPro/58.0__match_complete.xml` comes from https://ftp.ebi.ac.uk/pub/databases/interpro/releases/58.0/

All around the code, when we refer to "TASK_DATASET_PATH" we usually mean one of the subfolders of `data/processed/task_datasets`.

The `data/processed`directory should contain the output of processing made on the `data/raw` files using the scripts in `src/data_processing`. It should also contain protein embeddings generated using the script provided by Meta (described later).

**[Here](https://liveunibo-my.sharepoint.com/:u:/g/personal/marcello_fuschi_studio_unibo_it/EWfa8C1OC15DoL76bw0PP98BLYxh7oXkVTqfonEZ3drLfg?e=ouotUi "Here") you can download the already-processed data for the "2016" dataset, so that you can run the experiments immediately instead of having to process the raw data again. Instructions on how to run the experiments are provided below in this file.**

Once you have downloaded the processed dataset from the above paragraph, paste it into the `data/processed/task_datasets` directory.

### "2016" dataset

In `data/raw/task_datasets/2016`. Downloaded from https://github.com/bio-ontology-research-group/deepgoplus.
In the DeepGOPlus paper, it was the dataset that's used to compare DeepGOPlus with GOLabeler and DeepText2GO.

This same dataset is also used by other good-performing papers:
- PANDA2
- DeepGOPlus
- [DeepText2GO](https://www.sciencedirect.com/science/article/pii/S1046202318300021#s0030)
- GoLabeler

The .pkl files are pickle files containing a pandas dataframe.

### NetGO2 dataset

Taken from [DeepGOZero](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i238/6617515). Our train is their train+valid, and the testing set is the same as testing set in both DeepGOZero and NetGO2 papers. Currently, the `data/processed/task_datasets/NetGO2/all_proteins.fasta` file is incomplete because not all proteins in the dataset were found by the Uniprot mapping tool. This may be fixable if we find some other source.

## Methods

To encode proteins in an embedding vector, we use **ESM2**, a protein language model by Facebook. It first embeds amino acids, then you may take the average between them as the protein embedding.
Meta provided a script for embedding the proteins from a FASTA file, that we copied into `src/utils/embed_proteins_from_fasta`. Usage is described [here](https://github.com/facebookresearch/esm). 

Example (to embed proteins): `python src/utils/embed_proteins_from_fasta.py esm2_t33_650M_UR50D data/raw/CAFA3_training_data/uniprot_sprot_exp.fasta data/processed/CAFA3_training_data/protein_embeddings --include mean`

Names of the available ESM2 models: https://huggingface.co/facebook/esm2_t33_650M_UR50D

The solution we propose is a stacked ensemble model that uses multiple components:
- **Naive**: it assigns GO terms based on the frequency of annotations observed in the training dataset
- **DiamondScore**: it uses the Diamond tool to search for database proteins based on a query sequence, the results are accompanied by a bitcore: the final score is calculated by normalizing the sum of bitscores from all similar sequences returned by Diamond
- **InteractionScore**: the classifier exploits PPIs from the STRING network: Given a query protein _q_, its functional prediction score for a GO term _f_ is influenced by the annotations of proteins that interact with _q_
- **EmbeddingSimilarityScore**: a _k_-nearest neighbor search is performed operating cosine similarity between the query and training sequences and the final result is for a GO term _f_ is computed similarly to the DiamonScore (except the cosine similarities are re-weighted for better performance for high _k_)
- **FC on protein embeddings**
- **GNN on STRING PPIs**: it maps the STRING input graph (where each protein is represented by its own _embedding_) to a score for each GO term 


### Current best results on 2016 dataset

Stacked ensemble with 5 of the 6 components **(GNN was excluded)**, using **ESM2 15B** sequence embeddings:
- **MFO** | F_max: 0.594 (optimal threshold=0.19) | S_min: 8.651 | AUPR: 0.534
- **BPO** | F_max: 0.493 (optimal threshold=0.32) | S_min: 33.050 | AUPR: 0.426
- **CCO** | F_max: 0.722 (optimal threshold=0.33) | S_min: 7.138 | AUPR: 0.728

Stacked ensemble with just 4 components **(no FC-on-embeddings nor GNN)**, using **ESM2 15B** sequence embeddings:
- **MFO** | F_max: 0.591 (optimal threshold=0.18) | S_min: 8.617 | AUPR: 0.537
- **BPO** | F_max: 0.493 (optimal threshold=0.31) | S_min: 32.820 | AUPR: 0.434
- **CCO** | F_max: 0.718 (optimal threshold=0.32) | S_min: 7.198 | AUPR: 0.737

## How to run the solution

Since the solution is an ensemble method, there are multiple components to it. Each one of these base models can be run independently, to measure its performance. Provided that the `data` directory already contains all the necessary files (described in the sections above), you can run the experiments using the following commands:

- Naive: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/naive/demo.py`
- DiamondScore: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/diamondscore/demo.py`
- InteractionScore: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/interactionscore/demo.py`
- EmbeddingSimilarityScore: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/embeddingsimilarityscore/main.py`
- FC on embeddings: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/FC_on_embeddings/main.py`
- GNN on PPI & embeddings: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/GNN_on_PPI_with_embeddings/main.py`
- **Ensemble method**: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/stacked_ensemble/demo.py`

By default, the ensemble method only includes 4 of the 6 base models. To include all, toggle the `USE_ALL_COMPONENTS` boolean variable in `src/solution/stacked_ensemble/demo.py`.


## Relevant papers

**CAFA:**
- [CAFA1](http://www.ncbi.nlm.nih.gov/pubmed/23353650)
- [CAFA2](http://www.ncbi.nlm.nih.gov/pubmed/27604469)
- [CAFA3](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1835-8)
