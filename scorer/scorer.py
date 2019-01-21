import re
from functools import lru_cache, reduce
from operator import mul
from random import shuffle

from data.reader import DataReader
from utils.star import star


@lru_cache(maxsize=None)
def get_relations(plan: str):
    return list(re.findall("(<|>) (.*?) \[", plan))


class Scorer:
    is_trainable = False

    def eval(self, reader: DataReader):
        rank_scores = []
        for d in reader.data:
            plan = d.plan
            plans = d.plans
            shuffle(plans)
            plans = plans[:99999]
            plans.append(plan)
            plans = list(map(lambda s, p: p, sorted(
                zip([self.score(p) for p in plans], plans), key=star(lambda s, p: s), reverse=True)))

            rank_scores.append(plans.index(plan) / len(plans))

        return pow(reduce(mul, rank_scores, 1), 1 / len(rank_scores))

    def score(self, plan: str):
        raise NotImplementedError("Scorer.eval is not implemented")

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        raise NotImplementedError("Scorer.learn is not implemented")
