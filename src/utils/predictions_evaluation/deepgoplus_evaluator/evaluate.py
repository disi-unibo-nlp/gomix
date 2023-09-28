import os
from collections import deque, Counter
import math
import numpy as np
import json
import os

TASK_DATASET_PATH = os.environ["TASK_DATASET_PATH"]
assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'

# TODO: Be careful: it's probably necessary to use propagated annotations (also for test set). Double-check by seeing how the S_min mesaure is computed.
PROPAGATED_ANNOTATIONS_DIR = os.path.join(TASK_DATASET_PATH, 'propagated_annotations')

# Code adapted from https://github.com/bio-ontology-research-group/deepgoplus

ONT_ROOTS = {
    'MFO': 'GO:0003674',
    'BPO': 'GO:0008150',
    'CCO': 'GO:0005575'
}
NAMESPACES = {
    'MFO': 'molecular_function',
    'BPO': 'biological_process',
    'CCO': 'cellular_component'
}


def evaluate(
    gene_ontology_file_path: str,
    predictions: dict,  # dict: Prot ID -> list of (GO term, score)
    ground_truth: dict,  # dict: Prot ID -> list of GO terms
):
    go_rels = _Ontology(gene_ontology_file_path, with_rels=True)
    _calculate_ontology_ic(go_rels)
    for ont in NAMESPACES.keys():
        go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
        go_set.remove(ONT_ROOTS[ont])

        fmax = 0.0
        optimal_f_threshold = 0.0
        smin = 1000.0
        precisions = []
        recalls = []
        for t in range(101):
            threshold = t / 100.0

            all_gt_labels = []
            all_preds = []
            for prot_id, gt_labels in ground_truth.items():
                gt_labels = set([term for term in gt_labels if term in go_set])
                all_gt_labels.append(gt_labels)

                preds = set([term for term, score in predictions[prot_id] if score >= threshold])
                for go_term in preds.copy():
                    preds |= go_rels.get_ancestors(go_term)
                preds &= go_set  # Very important: it removes all terms that are not in the ontology we're considering.
                all_preds.append(preds)

            fscore, prec, rec, s = _evaluate_annots(go=go_rels, real_annots=all_gt_labels, pred_annots=all_preds)
            precisions.append(prec)
            recalls.append(rec)
            if fmax < fscore:
                fmax = fscore
                optimal_f_threshold = threshold
            if smin > s:
                smin = s

        # Compute AUPR
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)

        print('\n' + ont)
        print(f'F_max: {fmax:.3f} (optimal threshold={optimal_f_threshold:.2f})')
        print(f'S_min: {smin:.3f}')
        print(f'AUPR: {aupr:.3f}')


def _calculate_ontology_ic(go_rels):
    annotations = []
    for file_name in ['train.json', 'test.json']:
        with open(os.path.join(PROPAGATED_ANNOTATIONS_DIR, file_name)) as f:
            annotations_subgroup = json.load(f).values()  # list of lists of GO terms
            annotations_subgroup = [set(x) for x in annotations_subgroup]
            annotations.extend(annotations_subgroup)
    go_rels.calculate_ic(annotations)  # Pass list of sets of GO terms


def _evaluate_annots(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = real_annots[i].intersection(pred_annots[i])
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s


class _Ontology(object):
    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set
