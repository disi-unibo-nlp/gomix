import sys
from pathlib import Path
import json
import random
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader
from src.solution.components.embeddingsimilarityscore.Learner import Learner
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_evaluator
import argparse

EMBEDDING_TYPES = ['sequence']


def _run_demo():
    """
    Example usage:
    python src/solution/components/embeddingsimilarityscore/main.py data/processed/task_datasets/2016/propagated_annotations/train.json data/processed/task_datasets/2016/annotations/test.json data/raw/task_datasets/2016/go.obo
    """
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("train_annotations_file_path", help="Path to train annotations file")
    parser.add_argument("test_annotations_file_path", help="Path to test annotations file")
    parser.add_argument("gene_ontology_file_path", help="Path to gene ontology file")
    args = parser.parse_args()

    with open(args.train_annotations_file_path, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    with open(args.test_annotations_file_path, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    learner = Learner(
        train_annotations=train_annotations,
        prot_embedding_loader=ProteinEmbeddingLoader(types=EMBEDDING_TYPES)
    )

    print('Generating predictions...')
    predictions = {
        prot_id: [(go_term, score) for go_term, score in learner.predict(prot_id).items()]
        for prot_id in tqdm(test_annotations.keys())
    }

    print(f'Evaluating EmbeddingSimilarityScore predictions (with embedding type = {EMBEDDING_TYPES})...')
    evaluate_with_deepgoplus_evaluator(
        gene_ontology_file_path=args.gene_ontology_file_path,
        predictions=predictions,
        ground_truth=test_annotations
    )


if __name__ == '__main__':
    _run_demo()
