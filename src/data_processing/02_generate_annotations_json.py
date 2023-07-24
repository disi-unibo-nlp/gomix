import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import json
from glob import glob
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.GeneOntologyDAG import GeneOntologyDAG


def main(input_dir, output_dir):
    # Assuming there is only one .obo file in the directory
    gene_ontology_file_path = glob(os.path.join(input_dir, '*.obo'))[0]
    gene_ontology = GeneOntologyDAG(gene_ontology_file_path)

    pkl_files = glob(os.path.join(input_dir, '*_data.pkl'))

    for input_file in pkl_files:
        df = pd.read_pickle(input_file)

        # Remove obsolete terms
        df['annotations'] = df['annotations'].apply(lambda terms: [term for term in terms if not gene_ontology.is_obsolete(term)])
        df = df[df['annotations'].apply(lambda terms: len(terms) > 0)]

        # Extract the protein_id from the 'accessions' column as the key for data_dict
        df['protein_id'] = df['accessions'].apply(lambda x: x.split(';')[0])

        data_dict = dict(zip(df["protein_id"], df["annotations"]))

        basename = os.path.basename(input_file)
        filename, _ = os.path.splitext(basename)
        filename = filename.replace('_data', '')  # remove '_data' from filename
        output_filename = filename + '.json'
        output_file_path = os.path.join(output_dir, output_filename)

        with open(output_file_path, 'w') as json_file:
            json.dump(data_dict, json_file)

        print(f"File saved at {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to directory containing .pkl files and .obo file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
