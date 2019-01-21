from collections import defaultdict, Counter
from typing import List

from scorer.product_of_experts import Expert
from scorer.scorer import get_relations


class GlobalDirectionExpert(Expert):
    def __init__(self, plans: List[str]):
        direction = defaultdict(Counter)

        for plan in plans:
            matches = get_relations(plan)
            forward = len(list(filter(lambda a: a[0] == ">", matches)))

            direction[len(matches)][forward / (len(matches) + 1)] += 1

        self.probs = {}
        for e, c in direction.items():
            total = sum([x for x in c.values()]) + 1  # Smoothing
            self.probs[e] = {p: n / total for p, n in c.items()}
            self.probs[e]["UNK"] = 1 / total

    def eval(self, plan: str):
        matches = get_relations(plan)
        forward = len(list(filter(lambda a: a[0] == ">", matches)))

        relations = len(matches)
        direction = forward / (relations + 1)

        if relations not in self.probs:
            return 1  # Never encountered such size

        if direction not in self.probs[relations]:
            return self.probs[relations]["UNK"]  # Never encountered such percentage

        return self.probs[relations][direction]
