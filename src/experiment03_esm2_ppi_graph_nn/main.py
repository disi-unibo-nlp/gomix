import pickle
import os
import numpy as np
import torch
import torch.optim as optim
from ProteinGraphBuilder import ProteinGraphBuilder
from Net import Net
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

PPI_FILE_PATH = '../../data/processed/CAFA3_training_data/protein_representation/STRING_v11.0_network.json'
PROT_ANNOTATIONS_FILE_PATH = '../../data/processed/CAFA3_training_data/protein_propagated_annotations.json'

device = torch.device('mps')


def main():
    graph, num_classes = _build_or_load_graph()
    print('Protein graph:', graph)
    print(f'It contains {graph.num_nodes} nodes (proteins).')

    train_mask = _make_train_mask(graph.num_nodes, proportion_true=0.8)
    train_loader = NeighborLoader(graph, num_neighbors=[10, 5], batch_size=128, input_nodes=train_mask)

    test_mask = ~train_mask
    test_loader = NeighborLoader(graph, num_neighbors=[10, 5], batch_size=128, input_nodes=test_mask)

    print('Train-test split: {} - {}'.format(train_mask.sum(), test_mask.sum()))

    print(f'Using device: {device}')
    model = Net(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print('Training...')
    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
        model.train()

        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for batch in train_loader:
            batch.to(device)

            optimizer.zero_grad()
            targets = batch.y[:batch.batch_size]
            preds = model(batch)[:batch.batch_size]
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch.batch_size
            total_examples += batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        loss = total_loss / total_examples
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

        test_loss, performances_by_threshold = _evaluate(model, test_loader, loss_fn)
        print(f'Epoch {epoch:02d}, Test Loss: {test_loss:.4f}')

        f_max = 0
        opt_threshold = 0
        for threshold, (precision, recall) in performances_by_threshold.items():
            if precision + recall > 0:  # Avoid division by zero
                f1_score = 2 * precision * recall / (precision + recall)
                if f1_score > f_max:
                    f_max = f1_score
                    opt_threshold = threshold
        print(f'[{epoch:02d}, test,t={opt_threshold}] F_max: {f_max:.4f} (at optimal threshold)')

        print('——')
        scheduler.step()


def _build_or_load_graph():
    pickle_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/cache/experiment03', 'protein_graph.pickle')
    if os.path.exists(pickle_file_path):
        print("Loading graph from pickle file.")
        with open(pickle_file_path, 'rb') as file:
            return pickle.load(file)

    graph_builder = ProteinGraphBuilder(ppi_file_path=PPI_FILE_PATH)
    num_classes = graph_builder.load_targets(PROT_ANNOTATIONS_FILE_PATH)['num_classes']
    graph: GeometricData = graph_builder.build()
    graph.validate(raise_on_error=True)

    result = (graph, num_classes)

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(result, file)

    return result


def _make_train_mask(total: int, proportion_true=0.8):
    n_true = int(proportion_true * total)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[:n_true] = True
    mask = mask[torch.randperm(total)]
    return mask


@torch.no_grad()
def _evaluate(model, test_loader, loss_fn: torch.nn.Module):
    pbar = tqdm(total=len(test_loader.dataset))
    pbar.set_description('Evaluating')

    model.eval()
    total_loss = total_examples = 0
    all_preds = []
    all_targets = []
    for batch in test_loader:
        batch.to(device)

        targets = batch.y[:batch.batch_size]
        preds = model(batch)[:batch.batch_size]
        loss = loss_fn(preds, targets)
        total_loss += float(loss) * batch.batch_size
        total_examples += batch.batch_size

        all_preds.append(torch.sigmoid(preds).cpu())
        all_targets.append(targets.cpu())

        pbar.update(batch.batch_size)
    pbar.close()

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    performances_by_threshold = {}

    for threshold in np.round(np.arange(0.05, 0.75, 0.05), 2):
        polarized_preds = (all_preds >= threshold).float()
        true_positives = (polarized_preds * all_targets).sum(dim=1)
        false_positives = (polarized_preds * (1 - all_targets)).sum(dim=1)

        precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
        recall = true_positives.sum() / all_targets.sum()

        performances_by_threshold[threshold] = (precision, recall)

    loss = total_loss / total_examples
    return loss, performances_by_threshold


if __name__ == '__main__':
    main()
