#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct STRING PPI network."""
import json
from collections import defaultdict


def string2uniprot(file_path):
    """Map STRING accession to UniProt accession.
    :param file_path: path to mapping file
    :return: dict, key: STRING accession, value: UniProt accession
        { string_ac1: uniprot_ac1, string_ac2: uniprot_ac2, ... }
    """
    mapping = dict()
    with open(file_path) as fp:
        uniprot_to_string = json.load(fp)
        for uniprot_ac, string_ac in uniprot_to_string.items():
            mapping[string_ac] = uniprot_ac
    return mapping


def get_string_network(path_to_network, path_to_mapping):
    """Construct STRING PPI network.
    :param path_to_network: path to STRING network data
    :param path_to_mapping: path to mapping file
    :return: dict, PPI network
        { protein1: { protein1a: score1a, protein1b: score1b, ... },
          protein2: { protein2a: score2a, protein2b: score2b, ... },
          ... }
    """
    network = defaultdict(dict)
    mapping = string2uniprot(path_to_mapping)
    with open(path_to_network) as fp:
        fp.seek(0)
        for line in fp:
            if line.startswith("protein1"):
                continue
            string_ac1, string_ac2, score = line.strip().split()
            try:    # if no matched accession found, pass it
                protein1 = mapping[string_ac1]
                protein2 = mapping[string_ac2]
            except KeyError:
                continue
            score = float(score) / 1000
            network[protein1][protein2] = network[protein2][protein1] = score
    print(len(network), 'proteins in processed STRING network')
    return network


if __name__ == "__main__":
    network = get_string_network(
        path_to_network="../../../../data/raw/STRING/9606.protein.links.v11.5.txt",
        path_to_mapping="../../../../data/processed/CAFA3_training_data/protein_representation/uniprot_to_STRING_protein_mapping.json"
    )

    with open('../../../../data/processed/CAFA3_training_data/protein_representation/STRING_v11.5_network.json', 'w') as fp:
        json.dump(network, fp, indent=2)
