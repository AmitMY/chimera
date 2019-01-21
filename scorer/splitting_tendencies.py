from collections import defaultdict, Counter
from typing import List

from scorer.product_of_experts import Expert
from scorer.scorer import get_relations


class SplittingTendenciesExpert(Expert):
    def __init__(self, plans: List[str]):
        split = defaultdict(Counter)

        for plan in plans:
            matches = get_relations(plan)
            split[len(matches)][self.split(plan)] += 1

        self.probs = {}
        for e, c in split.items():
            total = sum([x for x in c.values()]) + 1  # Smoothing
            self.probs[e] = {p: n / total for p, n in c.items()}
            self.probs[e]["UNK"] = 1 / total

    def split(self, plan):
        return "-".join([str(len(get_relations(p))) for p in plan.split(".")])

    def eval(self, plan: str):
        matches = get_relations(plan)
        split = self.split(plan)

        relations = len(matches)

        if relations not in self.probs:
            return 1  # Never encountered such size

        if split not in self.probs[relations]:
            return self.probs[relations]["UNK"]  # Never encountered such split

        return self.probs[relations][split]
