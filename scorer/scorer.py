import re
from functools import reduce, lru_cache
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
        return reduce(mul, [e.eval(plan) for e in self.experts], 1)
