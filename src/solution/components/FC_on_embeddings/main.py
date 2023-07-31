from ProteinToGOModel import ProteinToGOModel
import torch
import json
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.utils.EmbeddedProteinsDataset import EmbeddedProteinsDataset
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_method
from src.utils.load_protein_embedding import load_protein_embedding
torch.manual_seed(0)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PROPAGATED_TRAIN_ANNOTS_FILE_PATH = os.path.join(THIS_DIR, '../../../../data/processed/task_datasets/2016/propagated_annotations/train.json')
OFFICIAL_TEST_ANNOTS_FILE_PATH = os.path.join(THIS_DIR, '../../../../data/processed/task_datasets/2016/annotations/test.json')
ALL_PROTEIN_EMBEDDINGS_DIR = os.path.join(THIS_DIR, '../../../../data/processed/task_datasets/2016/all_protein_embeddings/esm2_t48_15B_UR50D')
GENE_ONTOLOGY_FILE_PATH = os.path.join(THIS_DIR, '../../../../data/raw/task_datasets/2016/go.obo')

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

PROT_EMBEDDING_SIZE = 5120  # Number of elements in a single protein embedding vector


def main():
    dataset = _make_training_dataset()
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    print(f"Using device: {device}")

    model = ProteinToGOModel(protein_embedding_size=PROT_EMBEDDING_SIZE, output_size=len(dataset.go_term_to_index))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_f_max = -np.inf
    best_epoch = 0
    for epoch in range(80):
        print(f"Epoch {epoch+1}: Learning rate = {optimizer.param_groups[0]['lr']}")
        model.train()
        train_loss = 0.0
        for i, (prot_embeddings, targets) in enumerate(train_dataloader):
            prot_embeddings = prot_embeddings.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(prot_embeddings)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 200}')
                train_loss = 0.0

        val_loss, performances_by_threshold = _evaluate_for_validation(model, val_dataloader, loss_fn)
        print(f'[{epoch + 1}, validation] val_loss: {val_loss:.4f}')

        f_max = 0
        opt_threshold = 0
        for threshold, (precision, recall) in performances_by_threshold.items():
            if precision + recall > 0:  # Avoid division by zero
                f1_score = 2 * precision * recall / (precision + recall)
                if f1_score > f_max:
                    f_max = f1_score
                    opt_threshold = threshold
        print(f'[{epoch + 1}, validation] F_max: {f_max:.4f} (at optimal threshold t={opt_threshold})')

        if f_max > best_val_f_max:
            best_val_f_max = f_max
            best_epoch = epoch
        elif epoch - best_epoch > 3:  # Early stopping.
            print(f'Early stopping. Best F_max score on validation set was {best_val_f_max:.4f} at epoch {best_epoch}')
            break

        print('——')
        scheduler.step()

    print('Training finished. Let\'s now evaluate on test set (with the official criteria).')
    _evaluate_for_testing_with_official_criteria(model, go_term_to_index=dataset.go_term_to_index)


def _make_training_dataset():
    with open(PROPAGATED_TRAIN_ANNOTS_FILE_PATH, 'r') as f:
        train_annotations = json.load(f)

    return EmbeddedProteinsDataset(
        annotations=train_annotations,
        embeddings_dir=ALL_PROTEIN_EMBEDDINGS_DIR,
        go_term_to_index=_make_go_term_vocabulary(train_annotations)
    )


def _make_go_term_vocabulary(annotations):
    go_terms = set()
    for _, ann_go_terms in annotations.items():
        go_terms.update(ann_go_terms)
    go_terms = list(go_terms)
    return {go_term: i for i, go_term in enumerate(go_terms)}


def _evaluate_for_validation(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for prot_embeddings, targets in dataloader:
            prot_embeddings = prot_embeddings.to(device)
            targets = targets.to(device)
            outputs = model(prot_embeddings)
            running_loss += loss_fn(outputs, targets).item()
            all_preds.append(torch.sigmoid(outputs))
            all_targets.append(targets)
    running_loss /= len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    performances_by_threshold = {}

    for threshold in np.round(np.arange(0.01, 0.6, 0.01), 2):
        polarized_preds = (all_preds >= threshold).float()
        true_positives = (polarized_preds * all_targets).sum(dim=1)
        false_positives = (polarized_preds * (1 - all_targets)).sum(dim=1)

        precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
        recall = true_positives.sum() / all_targets.sum()

        performances_by_threshold[threshold] = (precision, recall)

    return running_loss, performances_by_threshold


def _evaluate_for_testing_with_official_criteria(model, go_term_to_index: dict):
    index_to_go_term = {v: k for k, v in go_term_to_index.items()}

    with open(OFFICIAL_TEST_ANNOTS_FILE_PATH, 'r') as f:
        test_annotations = json.load(f)
    prot_ids = list(test_annotations.keys())

    model.eval()
    with torch.no_grad():
        all_predictions = {}
        batch_size = 256
        for i in range(0, len(prot_ids), batch_size):
            batch_prot_ids = prot_ids[i:i+batch_size]
            batch_prot_embeddings = torch.stack([load_protein_embedding(ALL_PROTEIN_EMBEDDINGS_DIR, prot_id) for prot_id in batch_prot_ids])

            preds = model.predict(batch_prot_embeddings.to(device))
            top_scores, top_indices = torch.topk(preds, 200)  # Get the top k scores along with their indices

            for prot_id, scores, indices in zip(batch_prot_ids, top_scores, top_indices):
                all_predictions[prot_id] = [(index_to_go_term[idx.item()], score.item()) for score, idx in zip(scores, indices)]

    evaluate_with_deepgoplus_method(
        gene_ontology_file_path=GENE_ONTOLOGY_FILE_PATH,
        predictions=all_predictions,
        ground_truth=test_annotations
    )


if __name__ == '__main__':
    main()
