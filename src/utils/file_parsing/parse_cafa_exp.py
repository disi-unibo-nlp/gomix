def parse_cafa_exp_file(experimental_annotations_file):
    with open(experimental_annotations_file) as handle:
        for line in handle:
            accession_number, go_term, namespace = line.strip().split('\t')
            yield {
                'accession_number': accession_number,
                'go_term': go_term,
                'namespace': namespace
            }
