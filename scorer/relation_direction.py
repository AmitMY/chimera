from typing import List

import numpy as np

from scorer.product_of_experts import Expert
from scorer.scorer import get_relations


class RelationDirectionExpert(Expert):
    def __init__(self, plans: List[str]):
        matches = get_relations("\n".join(plans))

        direction = {}

        for match in matches:
            if match[1] not in direction:
                direction[match[1]] = {"f": 0, "b": 0}
            direction[match[1]]["f" if match[0] == ">" else "b"] += 1

        self.probs = {e: d["f"] / (d["f"] + d["b"]) for e, d in direction.items()}
        self.probs = {e: d if d != 1 else 0.999 for e, d in self.probs.items()}
        self.probs = {e: d if d != 0 else 0.001 for e, d in self.probs.items()}
        self.probs["UNK"] = np.mean(list(self.probs.values()))

    def eval(self, plan: str):
        matches = get_relations(plan)
        scores = []
        for match in matches:
            relation = "UNK" if match[1] not in self.probs else match[1]
            scores.append(self.probs[relation] if match[0] == ">" else (1 - self.probs[relation]))
        return scores
