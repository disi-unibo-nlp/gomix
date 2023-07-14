from goatools.obo_parser import GODag


class GeneOntologyDAG:
    def __init__(self):
        self.go_dag = GODag('./data/raw/GO/go-basic.obo', 'relationship')

    def is_obsolete(self, go_id: str) -> bool:
        return go_id not in self.go_dag

    def get_ancestors_ids(self, go_id: str) -> set:
        parents_ids = self._get_parents_ids(go_id)
        results = parents_ids.copy()
        for p_id in parents_ids:
            results.update(self.get_ancestors_ids(p_id))
        return results

    def _get_parents_ids(self, go_id: str) -> set:
        if go_id not in self.go_dag:
            # go_id is obsolete
            return set()
        parents = self.go_dag[go_id].parents
        return set([p.id for p in parents])
