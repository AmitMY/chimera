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

# REG
# # Bert REG
ExperimentsPipeline.enqueue("naive-reg", "Train Naive Referring Expressions Generator",
                            REGPipeline.mutate({"config": Config(reg=NaiveREG)}))
# # Naive REG
ExperimentsPipeline.enqueue("bert-reg", "Train BERT Referring Expressions Generator",
                            REGPipeline.mutate({"config": Config(reg=BertREG)}))

PostProcessPipeline = Pipeline()
PostProcessPipeline.enqueue("post-process", "Post process translations",
                            lambda f, x: x["translate"].copy().post_process(x[x["reg-name"] + "-reg"]))
PostProcessPipeline.enqueue("ents-reg-map", "Ents REG map",
                            lambda f, x: f["post-process"].ents_reg_map
                            if hasattr(f["post-process"], "ents_reg_map") else {})
PostProcessPipeline.enqueue("for-manual-eval", "Build manual eval file",
                            lambda f, x: json.dumps(f["post-process"].for_manual_evaluation()), ext="json")
PostProcessPipeline.enqueue("bleu", "Get BLEU score",
                            lambda f, x: f["post-process"].evaluate())

TranslatePipeline = Pipeline()
TranslatePipeline.enqueue("translate", "Translate plans",
                          lambda f, x: x[x["planner-name"] + "-planner"]["test-corpus"].copy()
                          .translate_plans(x["train-model"],
                                           x[x["planner-name"] + "-planner"]["train-planner"],
                                           x["test-config"]))
TranslatePipeline.enqueue("export", "Export plans and translations",
                          lambda f, x: json.dumps(f["translate"].export()), ext="json")
TranslatePipeline.enqueue("coverage", "Coverage of translation",
                          lambda f, x: f["translate"].coverage())
TranslatePipeline.enqueue("retries", "Retries of translation",
                          lambda f, x: f["translate"].retries())
TranslatePipeline.enqueue("eval-naive-reg", "Evaluate naive REG", PostProcessPipeline.mutate({"reg-name": "naive"}))
TranslatePipeline.enqueue("eval-bert-reg", "Evaluate BERT REG", PostProcessPipeline.mutate({"reg-name": "bert"}))

PlannerTranslatePipeline = Pipeline()
best_out_config = {"beam_size": 5, "find_best": False}
PlannerTranslatePipeline.enqueue("translate-best", "Translate best out",
                                 TranslatePipeline.mutate({"test-config": best_out_config}))
verify_out_config = {"beam_size": 5, "find_best": True}
PlannerTranslatePipeline.enqueue("translate-verify", "Translate verified out",
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

    all_res = [
        ExperimentsPipeline.mutate({"config": config}).execute("WebNLG Experiments", cache_name="exp/WebNLG0"),
        # ExperimentsPipeline.mutate({"config": config}).execute("WebNLG Experiments", cache_name="exp/WebNLG1"),
        # ExperimentsPipeline.mutate({"config": config}).execute("WebNLG Experiments", cache_name="exp/WebNLG2"),
        # ExperimentsPipeline.mutate({"config": config}).execute("WebNLG Experiments", cache_name="exp/WebNLG3"),
        # ExperimentsPipeline.mutate({"config": config}).execute("WebNLG Experiments", cache_name="exp/WebNLG4")
    ]

    for model_name in ["model", "model-feats"]:
        print(model_name)
        for decoding_method in ["best", "verify"]:
            # print("\t", decoding_method)
            all_retries = {"seen": 0, "unseen": 0}
            for res in all_res:
                retries = res[model_name]["translate-neural"]["translate-" + decoding_method]["retries"]
                all_retries["seen"] += retries["seen"]
                all_retries["unseen"] += retries["unseen"]
                # print("\t\t", retries)
            all_retries["seen"] = all_retries["seen"] / len(all_res)
            all_retries["unseen"] = all_retries["unseen"] / len(all_res)
            print("\t", decoding_method, all_retries)

    # print(res["naive-planner"]["test-corpus"].data[100].plan)
    # print(res["model"]["translate-naive"]["translate-best"]["translate"].data[100].plan)
    # print(res["model"]["translate-naive"]["translate-best"]["translate"].data[100].hyp)
    #
    # print()
    #
    # print(res["neural-planner"]["test-corpus"].data[100].plan)
    # print(res["model"]["translate-neural"]["translate-best"]["translate"].data[100].plan)
    # print(res["model"]["translate-neural"]["translate-best"]["translate"].data[100].hyp)

    for i, res in enumerate(all_res):
        print("\n\n\n", i)
        # Coverage
        table = []
        for model_name in ["model", "model-feats"]:
            model = res[model_name]
            print(model_name)
            for planner_name in ["naive", "neural"]:
                print("\t", planner_name)
                translation = model["translate-" + planner_name]

                for decoding_method in ["best", "verify"]:
                    cov = translation["translate-" + decoding_method]["coverage"]
                    # print("\t\t", decoding_method, "\t", translation["translate-" + decoding_method]["coverage"])
                    tabbed = "\t".join([str(round(a * 100, 1)) for a in
                                        [cov["seen"]["entities"], cov["seen"]["order"], cov["unseen"]["entities"],
                                         cov["unseen"]["order"]]])
                    table.append(tabbed)
                    print("\t\t", decoding_method, "\t", tabbed)
        print("\n".join(table))

        # BLEU
        bleu_table = []
        for model_name in ["model", "model-feats"]:
            model = res[model_name]
            print(model_name)
            for decoding_method in ["best", "verify"]:
                bleus = []

                for planner_name in ["naive", "neural"]:
                    translation = model["translate-" + planner_name]["translate-" + decoding_method]
                    for reg_name in ["naive", "bert"]:
                        bleus.append(translation["eval-" + reg_name + "-reg"]["bleu"][0])

                tabbed = "\t".join([str(round(a, 2)) for a in bleus])
                bleu_table.append(tabbed)

                print("\t", decoding_method, "\t", tabbed)
        print("\n".join(bleu_table))
