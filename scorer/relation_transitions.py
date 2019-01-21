from collections import defaultdict, Counter
from typing import List

from scorer.product_of_experts import Expert
from scorer.scorer import get_relations


class RelationTransitionsExpert(Expert):
    def __init__(self, plans: List[str]):
        adjacent = defaultdict(Counter)

        for plan in plans:
            for p in plan.split("."):
                matches = get_relations(p)

                for i in range(len(matches) - 1):
                    adjacent[matches[i][1]][matches[i + 1][1]] += 1
                if len(matches) > 0:
                    adjacent[matches[-1][1]]["EOS"] += 1

        self.probs = {}
        for e, c in adjacent.items():
            total = sum([x for x in c.values()]) + 1  # Smoothing
            self.probs[e] = {p: n / total for p, n in c.items()}
            self.probs[e]["UNK"] = 1 / total

    def eval(self, plan: str):
        def get_prob(r1, r2):
            if r1 not in self.probs:
                return 1  # Never encountered such edge

            if r2 not in self.probs[r1]:
                return self.probs[r1]["UNK"]  # Never encountered such adjacency

            return self.probs[r1][r2]

        scores = []

        for p in plan.split("."):
            matches = get_relations(p)

            for i in range(len(matches) - 1):
                scores.append(get_prob(matches[i][1], matches[i + 1][1]))
            scores.append(get_prob(matches[-1][1], "EOS"))

        return scores
