import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import os
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from src.experiment02_ppi_and_gcn_and_dense.utils.get_ppi_similarity_matrix import get_ppi_similarity_matrix
from src.experiment02_ppi_and_gcn_and_dense.utils.get_protein_features_from_ppi_similarity_matrix import get_protein_features_from_ppi_similarity_matrix
from src.experiment02_ppi_and_gcn_and_dense.utils.ProteinToGOModel import ProteinToGOModel
from typing import List

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def main():
    # The PPI similarity matrix only contains proteins in the training data because that's how we built it.
    # For now, though, it doesn't contain ALL proteins in training data because we didn't have them readily available.
    ppi_similarity_matrix, sorted_prot_ids = _load_ppi_similarity_matrix()
    print('Shape of PPI similarity matrix:', ppi_similarity_matrix.shape)
    print('Number of proteins in the matrix:', len(sorted_prot_ids), '\n')

    # Initialize the protein features (= graph node features) to use as input in the model.
    protein_features = get_protein_features_from_ppi_similarity_matrix(ppi_similarity_matrix)
    print('Shape of protein features matrix:', protein_features.shape, '\n')

    # Load training annotations, i.e. full protein -> GO term mapping
    # Format: { protein1: [term1, term2], protein2: [term3] }
    with open('../../data/processed/CAFA3_training_data/protein_propagated_annotations.json', 'r') as f:
        prot_annotations = json.load(f)

    # Our model will output a tensor of numbers, i.e. predictions for each GO term.
    # This will be our reference list to translate from index to GO term.
    sorted_go_terms = _get_go_terms_in_annotations(prot_annotations)
    print('Number of GO terms in the vocabulary:', len(sorted_go_terms), '\n')

    # transform annotations to a DataFrame like:
    #           GOterm1   GOterm2   GOterm3
    # protein1     1         0         1
    # protein2     0         1         0
    # protein3     0         0         1
    #
    # These data will be used when training/testing the (protein -> GO term) prediction model.
    prot_annotations_df = _transform_prot_annotations_to_dataframe(prot_annotations, sorted_prot_ids, sorted_go_terms)
    print('Annotations DataFrame:')
    print(prot_annotations_df.head(), '\n')

    # Train the model.

    train_prot_ids, test_prot_ids = train_test_split(sorted_prot_ids, test_size=0.2, random_state=42)

    model = ProteinToGOModel(n_protein=protein_features.shape[0], n_term=len(sorted_go_terms))
    print(f'Moving model to {device}')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    print('Starting model training...')

    protein_features = torch.FloatTensor(protein_features).to(device)
    ppi_similarity_matrix = torch.FloatTensor(ppi_similarity_matrix).to(device)

    for epoch in range(10):
        model.train()

        optimizer.zero_grad()
        outputs = model(protein_features, ppi_similarity_matrix)
        batch_loss = _calc_predictions_loss(outputs, prot_annotations_df, train_prot_ids)
        batch_loss.backward()
        optimizer.step()

        print(f'[{epoch + 1}] loss: {batch_loss.item()}')


def _load_ppi_similarity_matrix():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, '../../data/cache/experiment02', 'ppi_similarity_matrix.pickle')
    if os.path.exists(pickle_file):
        print("Warning: Using cached PPI similarity matrix. Check if you made changes to the STRING network JSON file.")
        with open(pickle_file, "rb") as file:
            return pickle.load(file)

    result = get_ppi_similarity_matrix('../../data/processed/CAFA3_training_data/protein_representation/STRING_v11.5_network.json')

    with open(pickle_file, "wb") as file:
        pickle.dump(result, file)

    return result


def _get_go_terms_in_annotations(annotations):
    go_terms = set()
    for _, ann_go_terms in annotations.items():
        go_terms.update(ann_go_terms)
    return list(go_terms)


def _transform_prot_annotations_to_dataframe(prot_annotations, sorted_prot_ids, sorted_go_terms):
    mlb = MultiLabelBinarizer()
    prot_annotations_df = pd.DataFrame(
        mlb.fit_transform(prot_annotations.values()),
        index=prot_annotations.keys(),
        columns=mlb.classes_
    ).reindex(
        index=sorted_prot_ids,
        columns=sorted_go_terms,
        fill_value=0
    )
    return prot_annotations_df


def _calc_predictions_loss(
    preds: torch.Tensor,
    prot_annotations_df: pd.DataFrame,
    prot_ids_to_consider: List,  # For train-test split indication. Loss is calculated only on these proteins.
) -> torch.Tensor:
    device = preds.device

    criterion = torch.nn.BCEWithLogitsLoss()

    prot_annotations_to_consider = torch.FloatTensor(
        prot_annotations_df.loc[prot_ids_to_consider].values,
        device=device
    )

    losses = []
    for prot_id in prot_ids_to_consider:
        pred_index = prot_annotations_df.index.get_loc(prot_id)
        pred = preds[pred_index]  # Get the prediction for the current protein

        # Get the ground truth for the current protein
        gt = prot_annotations_to_consider[prot_ids_to_consider.index(prot_id)]

        loss = criterion(pred, gt)
        losses.append(loss)

    average_loss = torch.mean(torch.stack(losses))
    return average_loss


if __name__ == '__main__':
    main()
