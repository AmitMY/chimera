import random

from data.DelexWebNLG.reader import DelexWebNLGDataReader
from data.WebNLG.reader import WebNLGDataReader
from data.reader import DataReader, DataSetType
from process.evaluation import EvaluationPipeline
from process.learn_score import LearnScorePipeline
from process.pre_process import TrainingPreProcessPipeline, TestingPreProcessPipeline
from process.train_model import TrainModelPipeline
from process.translate import TranslatePipeline
from utils.pipeline import Pipeline


class Config:
    def __init__(self, reader: DataReader, test_reader: DataReader = None):
        self.reader = {
            DataSetType.TRAIN: reader,
            DataSetType.DEV: reader,
            DataSetType.TEST: test_reader if test_reader else reader,
        }


MainPipeline = Pipeline()
MainPipeline.enqueue("pre-process", "Pre-process training data", TrainingPreProcessPipeline)
MainPipeline.enqueue("train-model", "Train Model", TrainModelPipeline)
MainPipeline.enqueue("learn-score", "Learn Score", LearnScorePipeline)
MainPipeline.enqueue("test-corpus", "Pre-process test data", TestingPreProcessPipeline)
MainPipeline.enqueue("translate", "Translate Test", TranslatePipeline)
MainPipeline.enqueue("evaluate", "Evaluate Translations", EvaluationPipeline)

if __name__ == "__main__":
    # config = Config(reader=WebNLGDataReader)
    config = Config(reader=DelexWebNLGDataReader, test_reader=WebNLGDataReader)
    res = MainPipeline.mutate({"config": config}).execute("Main")

    # print()
    # g = Graph([["Elliot_See", "deathDate", '"1966-02-28"']])
    # plans = g.exhaustive_plan().linearizations()
    # for p in plans:
    #     print(res["learn-score"].eval(p), p)

    print()

    d = random.choice(res["translate"].data)
    print("Random Sample:")
    print("Graph:", d.graph.as_rdf())
    print("Plan:", d.plan)
    print("Translation:", d.hyp)
    print("Reference:  ", d.text)

    print()

    print("BLEU", res["evaluate"])
