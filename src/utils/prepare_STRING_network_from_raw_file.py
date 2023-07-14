import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm


def string2uniprot(file_path):
    mapping = dict()
    with open(file_path) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            entries = line.strip().split('\t')
            uniprot_ac = entries[1].split('|')[0]
            string_ac = entries[2]
            mapping[string_ac] = uniprot_ac
    return mapping


def get_string_network(path_to_network, path_to_mapping, protein_ids_to_keep):
    network = defaultdict(dict)
    mapping = string2uniprot(path_to_mapping)

    def count_lines_in_file(path):
        num_lines = 0
        with open(path, 'r') as fp:
            for _ in tqdm(fp, desc="Counting lines", unit="line"):
                num_lines += 1
                if '_' in globals():
                    del _
        return num_lines

    num_lines = count_lines_in_file(path_to_network)
    with open(path_to_network, 'r') as fp:
        for line in tqdm(fp, total=num_lines, unit='line', desc="Processing network file"):
            if line.startswith("protein1"):
                continue
            string_ac1, string_ac2, score = line.strip().split()
            try:  # if no matched accession found, pass it
                protein1 = mapping[string_ac1]
                protein2 = mapping[string_ac2]
            except KeyError:
                continue
            if protein1 in protein_ids_to_keep and protein2 in protein_ids_to_keep:
                score = float(score) / 1000
                network[protein1][protein2] = network[protein2][protein1] = score
    return network


def get_file_path_in_script_dir(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('protein_ids_file_path', type=str)
    parser.add_argument('network_file_path', type=str)
    parser.add_argument('mapping_file_path', type=str)
    args = parser.parse_args()

    with open(args.protein_ids_file_path, 'r') as f:
        protein_ids_to_keep = set(json.load(f))

    network = get_string_network(args.network_file_path, args.mapping_file_path, protein_ids_to_keep)

    # Write to output file
    with open(get_file_path_in_script_dir('processed_network.json'), 'w') as fp:
        json.dump(network, fp, indent=2)

    print(f"Number of proteins in the final adjacency matrix: {len(network.keys())}")
