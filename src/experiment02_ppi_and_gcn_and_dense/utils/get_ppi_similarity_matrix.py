import numpy as np
import pandas as pd
import json
from typing import Tuple, List


# Inspired by HPODNets paper.
def get_ppi_similarity_matrix(ppi_file_path: str) -> Tuple[np.ndarray, List[str]]:
    # load protein-protein similarity network
    # format: { protein1: { protein2: score2, protein3: score3, ... }, ... }
    with open(ppi_file_path) as fp:
        ppi = json.load(fp)

    # Get the unique protein IDs and sort them
    prot_ids = sorted(ppi.keys())

    # Create a DataFrame with protein IDs as both row and column indexes
    ppi_df = pd.DataFrame(ppi, index=prot_ids, columns=prot_ids).fillna(0)

    # Convert the DataFrame to a NumPy array
    ppi_array = ppi_df.values

    # compute negative half power of degree matrix which is a diagonal matrix
    row_sums = np.sum(ppi_array, 1)
    diag = np.zeros_like(row_sums)
    nonzero_rows = row_sums != 0
    diag[nonzero_rows] = 1 / np.sqrt(row_sums[nonzero_rows])
    neg_half_power_degree_matrix = np.diag(diag)

    # construct normalized similarity network
    normalized_ppi = np.matmul(np.matmul(neg_half_power_degree_matrix, ppi_array),
                               neg_half_power_degree_matrix)

    return normalized_ppi, prot_ids
