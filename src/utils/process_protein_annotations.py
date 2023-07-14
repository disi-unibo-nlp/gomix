import argparse
from src.utils.file_parsing.parse_cafa_exp import parse_cafa_exp_file
import json
from src.utils.GeneOntologyDAG import GeneOntologyDAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    gene_ontology = GeneOntologyDAG()

    prot_to_annotations = {}

    annotations = list(parse_cafa_exp_file(args.annotations_file))
    skipped_count = 0
    for ann in annotations:
        prot_id, ann_go_id = ann['accession_number'], ann['go_term']
        if gene_ontology.is_obsolete(ann_go_id):
            skipped_count += 1
            continue
        prot_to_annotations[prot_id] = prot_to_annotations.get(prot_id, set())\
            .union({ann_go_id})\
            .union(gene_ontology.get_ancestors_ids(ann_go_id))  # True-path-rule

    print(f'Skipped {skipped_count:,} annotations because they are obsolete.')
    print(f'Found {len(prot_to_annotations):,} proteins with annotations.')

    prot_to_annotations = {
        prot_id: list(go_ids)
        for prot_id, go_ids in prot_to_annotations.items()
    }
    with open(args.out_file, 'w') as f:
        json.dump(prot_to_annotations, f)


if __name__ == '__main__':
    main()
