import random

from process.evaluation import EvaluationPipeline
from process.learn_score import LearnScorePipeline
from process.pre_process import TrainingPreProcessPipeline, TestingPreProcessPipeline
from process.train_model import TrainModelPipeline
from process.translate import TranslatePipeline
from utils.pipeline import Pipeline

MainPipeline = Pipeline()
MainPipeline.enqueue("pre-process", "Pre-process training data", TrainingPreProcessPipeline)
MainPipeline.enqueue("train-model", "Train Model", TrainModelPipeline)
MainPipeline.enqueue("learn-score", "Learn Score", LearnScorePipeline)
MainPipeline.enqueue("test-corpus", "Pre-process test data", TestingPreProcessPipeline)
MainPipeline.enqueue("translate", "Translate Test", TranslatePipeline)
MainPipeline.enqueue("evaluate", "Evaluate Translations", EvaluationPipeline)

if __name__ == "__main__":
    res = MainPipeline.execute("Main")

    print()

    g, p, t, s = random.choice(res["translate"].data)
    print("Random Sample:")
    print("Graph:", g.as_rdf())
    print("Plan:", p)
    print("Translation:", t)
    print("Reference:  ", s)

    print()

    print("BLEU", res["evaluate"])
