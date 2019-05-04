from functools import reduce
from operator import mul

from data.reader import DataReader
from scorer.scorer import Scorer


class Expert:
    def eval(self, plan: str):
        raise NotImplementedError("Must implement eval")


class WeightedProductOfExperts(Scorer):
    def __init__(self, expert_constructors):
        self.expert_constructors = expert_constructors
        self.experts = []
        self.weights = []

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        if len(self.experts) == 0:
            plans = [d.plan for d in train_reader.data]
            self.experts = [e(plans) for e in self.expert_constructors]
            self.weights = [1 for _ in self.experts]

    def score(self, plan: str):
        scores = [e.eval(plan) for e in self.experts]
        scores = [pow(reduce(mul, s, 1), 1 / len(s)) if isinstance(s, list) else s for s in scores]
        return reduce(mul, scores, 1)
