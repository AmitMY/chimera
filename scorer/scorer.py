import re
from functools import reduce, lru_cache
from json import dumps
from operator import mul


@lru_cache(maxsize=None)
def get_relations(plan: str):
    return list(re.findall("(<|>) (.*?) \[", plan))


class Expert:
    def eval(self, plan: str):
        raise NotImplementedError("Must implement eval")


class ProductOfExperts:
    def __init__(self, experts):
        self.experts = experts

    def eval(self, plan: str):
        scores = [e.eval(plan) for e in self.experts]
        scores = [pow(reduce(mul, s, 1), 1 / len(s)) if isinstance(s, list) else s for s in scores]
        return reduce(mul, scores, 1)
