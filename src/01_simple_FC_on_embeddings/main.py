from src.utils.EmbeddedProteinsDataset import EmbeddedProteinsDataset
from ProteinToGOModel import ProteinToGOModel
import torch
import json
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

PROT_EMBEDDING_SIZE = 2560  # Number of elements in a single protein embedding vector


def main():
    dataset = _make_dataset_from_cafa3_training_data()
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    print(f"Using device: {device}")

    model = ProteinToGOModel(protein_embedding_size=PROT_EMBEDDING_SIZE, output_size=len(dataset.go_term_to_index))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(50):
        print(f"Epoch {epoch+1}: Learning rate = {optimizer.param_groups[0]['lr']}")
        model.train()
        train_loss = 0.0
        for i, (protein_embeddings, targets) in enumerate(train_dataloader):
            protein_embeddings = protein_embeddings.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(protein_embeddings)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 200}')
                train_loss = 0.0

        test_loss, performances_by_threshold = _evaluate(model, test_dataloader, loss_fn)
        print(f'[{epoch + 1}, test] test_loss: {test_loss:.4f}')
        for threshold, (precision, recall) in performances_by_threshold.items():
            f1_score = 2 * precision * recall / (precision + recall)
            print(f'[{epoch + 1}, test,t={threshold}] f1_score: {f1_score:.4f} (precision: {precision:.4f}, recall: {recall:.4f})')
        print('——')
        scheduler.step()


def _make_dataset_from_cafa3_training_data():
    with open('../../data/processed/CAFA3_training_data/protein_propagated_annotations.json', 'r') as f:
        annotations = json.load(f)

    return EmbeddedProteinsDataset(
        annotations=annotations,
        embeddings_dir='../../data/processed/CAFA3_training_data/protein_representation/protein_embeddings_esm2_t36_3B_UR50D',
        go_term_to_index=_make_go_term_vocabulary(annotations)
    )


def _make_go_term_vocabulary(annotations):
    go_terms = set()
    for _, ann_go_terms in annotations.items():
        go_terms.update(ann_go_terms)
    go_terms = list(go_terms)
    return {go_term: i for i, go_term in enumerate(go_terms)}


def _evaluate(model, dataloader, loss_fn):
    model.eval()
    test_loss = 0.0
    test_outputs = []
    test_targets = []
    with torch.no_grad():
        for protein_embeddings, targets in dataloader:
            protein_embeddings = protein_embeddings.to(device)
            targets = targets.to(device)
            outputs = model(protein_embeddings)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            test_outputs.append(torch.sigmoid(outputs))
            test_targets.append(targets)
    test_loss /= len(dataloader)
    test_outputs = torch.cat(test_outputs)
    test_targets = torch.cat(test_targets)

    performances_by_threshold = {}

    for threshold in [0.1, 0.25, 0.5]:
        predictions = (test_outputs > threshold).float()
        true_positives = (predictions * test_targets).sum(dim=1)
        false_positives = (predictions * (1 - test_targets)).sum(dim=1)

        precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
        recall = true_positives.sum() / test_targets.sum()

        performances_by_threshold[threshold] = (precision, recall)

    return test_loss, performances_by_threshold


if __name__ == '__main__':
    main()
