import numpy as np
from tqdm import tqdm

from data.reader import DataReader
from planner.planner import Planner
from scorer.scorer import Scorer
from utils.graph import Graph


class NaivePlanner(Planner):
    is_parallel = True
    re_plan = "PREMADE"

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        for i in range(5):
            self.scorer.learn(train_reader, dev_reader)
            if not self.scorer.is_trainable:
                break
        return self

    def score(self, g: Graph, plan: str):
        return self.scorer.score(plan)

    def plan_best(self, g: Graph, ranker_plans=None):
        if ranker_plans:
            all_plans = list(set(ranker_plans))
        else:
            all_plans = list(self.plan_all(g))

        plan_scores = [(p, self.scorer.score(p)) for p in tqdm(all_plans)]
        plan_scores = sorted(plan_scores, key=lambda a: a[1], reverse=True)

        best_50_plans = [p for p, s in plan_scores[:50]]

        return best_50_plans
