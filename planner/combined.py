import numpy as np
from data.reader import DataReader
from planner.planner import Planner
from scorer.scorer import Scorer
from utils.graph import Graph
from typing import Callable, Tuple, List


class CombinedPlanner(Planner):
    def __init__(self, planners: Tuple[Planner, Planner]):
        self.planners = planners

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        self.planners = [p.learn(train_reader, dev_reader) for p in self.planners]
        return self

    def plan_best(self, g: Graph, ranker_plans=None):
        if ranker_plans:
            raise NotImplementedError("Planner.plan_best is not implemented when ranker_plans is defined")

        ranker, re_ranker = self.planners

        ranker_plans = ranker.plan_random(g, 50) + [ranker.plan_best(g)]

        return re_ranker.plan_best(g, ranker_plans=ranker_plans)
