from data.reader import DataReader
from utils.graph import Graph


class Planner:
    is_parallel = False

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        raise NotImplementedError("Planner.learn is not implemented")

    def plan_best(self, g: Graph):
        raise NotImplementedError("Planner.plan_best is not implemented")

    def plan_all(self, g: Graph):
        raise NotImplementedError("Planner.plan_all is not implemented")
