import numpy as np

from data.reader import DataReader
from planner.planner import Planner
from scorer.scorer import Scorer
from utils.graph import Graph


class NaivePlanner(Planner):
    is_parallel = True

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        for i in range(5):
            self.scorer.learn(train_reader, dev_reader)
            if not self.scorer.is_trainable:
                break
        return self

    def plan_best(self, g: Graph, ranker_plans=None):
        if ranker_plans:
            all_plans = list(set(ranker_plans))
        else:
            all_plans = self.plan_all(g)

        if len(all_plans) == 0:
            return ""
        all_scores = [self.scorer.score(p) for p in all_plans]
        max_i = np.argmax(all_scores)

        return all_plans[max_i]

    def plan_all(self, g: Graph):
        return g.exhaustive_plan(force_tree=False).linearizations()
