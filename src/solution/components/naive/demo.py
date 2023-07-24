import os
import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.solution.components.naive.NaiveLearner import NaiveLearner
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_method
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


"""
Example usage:
python src/solution/components/naive/demo.py data/processed/task_datasets/2016/propagated_annotations/train.json data/processed/task_datasets/2016/annotations/test.json data/raw/task_datasets/2016/go.obo
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_annotations_file_path", help="Path to train annotations file")
    parser.add_argument("test_annotations_file_path", help="Path to test annotations file")
    parser.add_argument("gene_ontology_file_path", help="Path to gene ontology file")
    args = parser.parse_args()

    with open(args.train_annotations_file_path, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    with open(args.test_annotations_file_path, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    naive_learner = NaiveLearner(train_annotations)

    predictions = {
        prot_id: [(go_term, score) for go_term, score in naive_learner.predict().items()]
        for prot_id in test_annotations.keys()
    }

    print('Evaluating Naive predictions...')
    evaluate_with_deepgoplus_method(gene_ontology_file_path=args.gene_ontology_file_path, predictions=predictions, ground_truth=test_annotations)


if __name__ == '__main__':
    main()
