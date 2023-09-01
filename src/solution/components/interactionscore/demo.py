import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.solution.components.interactionscore.InteractionScoreLearner import InteractionScoreLearner
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_evaluator
import argparse


"""
Example usage:
python src/solution/components/interactionscore/demo.py data/processed/task_datasets/2016/propagated_annotations/train.json data/processed/task_datasets/2016/annotations/test.json data/raw/task_datasets/2016/go.obo data/processed/task_datasets/2016/all_proteins_STRING_interactions.json
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_annotations_file_path", help="Path to train annotations file")
    parser.add_argument("test_annotations_file_path", help="Path to test annotations file")
    parser.add_argument("gene_ontology_file_path", help="Path to gene ontology file")
    parser.add_argument("ppi_file_path")
    args = parser.parse_args()

    with open(args.train_annotations_file_path, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    with open(args.test_annotations_file_path, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    learner = InteractionScoreLearner(train_annotations, args.ppi_file_path)

    predictions = {
        prot_id: [(go_term, score) for go_term, score in learner.predict(prot_id).items()]
        for prot_id in test_annotations.keys()
    }

    print('Evaluating InteractionScore predictions...')
    evaluate_with_deepgoplus_evaluator(
        gene_ontology_file_path=args.gene_ontology_file_path,
        predictions=predictions,
        ground_truth=test_annotations
    )


if __name__ == '__main__':
    main()
