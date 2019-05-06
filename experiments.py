import copy
import json

from data.WebNLG.reader import WebNLGDataReader
from main import Config
from planner.naive_planner import NaivePlanner
from planner.neural_planner import NeuralPlanner
from process.pre_process import TrainingPreProcessPipeline, TestingPreProcessPipeline
from process.reg import REGPipeline
from process.train_model import TrainModelPipeline, DEFAULT_TRAIN_CONFIG, DEFAULT_TEST_CONFIG
from process.train_planner import TrainPlannerPipeline
from reg.bert import BertREG
from reg.naive import NaiveREG
from scorer.global_direction import GlobalDirectionExpert
from scorer.product_of_experts import WeightedProductOfExperts
from scorer.relation_direction import RelationDirectionExpert
from scorer.relation_transitions import RelationTransitionsExpert
from scorer.splitting_tendencies import SplittingTendenciesExpert
from utils.pipeline import Pipeline

naive_planner = NaivePlanner(WeightedProductOfExperts(
    [RelationDirectionExpert, GlobalDirectionExpert, SplittingTendenciesExpert, RelationTransitionsExpert]))
neural_planner = NeuralPlanner()

PlanPipeline = Pipeline()
PlanPipeline.enqueue("train-planner", "Train Planner", TrainPlannerPipeline)
PlanPipeline.enqueue("test-corpus", "Pre-process test data", TestingPreProcessPipeline)

ExperimentsPipeline = Pipeline()
ExperimentsPipeline.enqueue("pre-process", "Pre-process training data", TrainingPreProcessPipeline)

# Train all planners
# # Naive Planner
ExperimentsPipeline.enqueue("naive-planner", "Train Naive Planner",
                            PlanPipeline.mutate({"config": Config(reader=WebNLGDataReader, planner=naive_planner)}))
# # Neural Planner
ExperimentsPipeline.enqueue("neural-planner", "Train Neural Planner",
                            PlanPipeline.mutate({"config": Config(reader=WebNLGDataReader, planner=neural_planner)}))

# # REG
# # # Bert REG
# ExperimentsPipeline.enqueue("naive-reg", "Train Naive Referring Expressions Generator",
#                             REGPipeline.mutate({"config": Config(reg=NaiveREG)}))
# # # Naive REG
# ExperimentsPipeline.enqueue("bert-reg", "Train BERT Referring Expressions Generator",
#                             REGPipeline.mutate({"config": Config(reg=BertREG)}))

TranslatePipeline = Pipeline()
TranslatePipeline.enqueue("translate", "Translate plans",
                          lambda f, x: x[x["planner-name"] + "-planner"]["test-corpus"].copy()
                          .translate_plans(x["train-model"], x["test-config"]))
TranslatePipeline.enqueue("coverage", "Coverage of translation",
                          lambda f, x: f["translate"].coverage())

PlannerTranslatePipeline = Pipeline()
best_out_config = {"beam_size": 5, "find_best": False}
PlannerTranslatePipeline.enqueue("translate-best", "Translate best out",
                                 TranslatePipeline.mutate({"test-config": best_out_config}))
verify_out_config = {"beam_size": 5, "find_best": True}
PlannerTranslatePipeline.enqueue("translate-verify", "Translate best out",
                                 TranslatePipeline.mutate({"test-config": verify_out_config}))


def model_pipeline(train_config):
    pipeline = Pipeline()
    pipeline.enqueue("train-model", "Train Model",
                     TrainModelPipeline.mutate({"train-config": train_config, "test-config": DEFAULT_TEST_CONFIG}))
    pipeline.enqueue("translate-naive", "Translate Naive Plans",
                     PlannerTranslatePipeline.mutate({"planner-name": "naive"}))
    pipeline.enqueue("translate-neural", "Translate Neural Plans",
                     PlannerTranslatePipeline.mutate({"planner-name": "neural"}))
    return pipeline


# Train model
# # Train without features
no_feats_config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
no_feats_config["features"] = False
del no_feats_config["train"]["feat_vec_size"]
del no_feats_config["train"]["feat_merge"]
ExperimentsPipeline.enqueue("model", "Train Model without features", model_pipeline(no_feats_config))
# # Train with features
feats_config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
feats_config["features"] = True
ExperimentsPipeline.enqueue("model-feats", "Train Model with features", model_pipeline(feats_config))

if __name__ == "__main__":
    config = Config(reader=WebNLGDataReader)

    res = ExperimentsPipeline.mutate({"config": config}) \
        .execute("WebNLG Experiments", cache_name="WebNLG_Exp")

    # print(res["naive-planner"]["test-corpus"].data[100].plan)
    # print(res["model"]["translate-naive"]["translate-best"]["translate"].data[100].plan)
    # print(res["model"]["translate-naive"]["translate-best"]["translate"].data[100].hyp)
    #
    # print()
    #
    # print(res["neural-planner"]["test-corpus"].data[100].plan)
    # print(res["model"]["translate-neural"]["translate-best"]["translate"].data[100].plan)
    # print(res["model"]["translate-neural"]["translate-best"]["translate"].data[100].hyp)

    for model_name in ["model", "model-feats"]:
        model = res[model_name]
        print(model_name)
        for planner_name in ["naive", "neural"]:
            print("\t", planner_name)
            translation = model["translate-" + planner_name]

            for decoding_method in ["best", "verify"]:
                cov = translation["translate-" + decoding_method]["coverage"]
                print("\t\t", decoding_method, "\t", translation["translate-" + decoding_method]["coverage"])
                tabbed = ",\t".join([str(round(a*100, 1)) for a in
                                    [cov["seen"]["entities"], cov["seen"]["order"], cov["unseen"]["entities"],
                                     cov["unseen"]["order"]]])

                print("\t\t", decoding_method, "\t", tabbed)
