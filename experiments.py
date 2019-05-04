from data.WebNLG.reader import WebNLGDataReader
from main import Config
from planner.naive_planner import NaivePlanner
from planner.neural_planner import NeuralPlanner
from process.pre_process import TrainingPreProcessPipeline, TestingPreProcessPipeline
from process.train_model import TrainModelPipeline, DEFAULT_TRAIN_CONFIG
from process.train_planner import TrainPlannerPipeline
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

# Train model
# # Train without features
no_feats_config = DEFAULT_TRAIN_CONFIG.copy()
no_feats_config["features"] = False
del no_feats_config["train"]["feat_vec_size"]
del no_feats_config["train"]["feat_merge"]
ExperimentsPipeline.enqueue("train-model-no-feats", "Train Model", TrainModelPipeline.mutate({"config": no_feats_config}))
# # Train with features
feats_config = DEFAULT_TRAIN_CONFIG.copy()
feats_config["features"] = True
ExperimentsPipeline.enqueue("train-model-feats", "Train Model", TrainModelPipeline.mutate({"config": feats_config}))
#
#
# ExperimentsPipeline.enqueue("train-reg", "Train Referring Expressions Generator", REGPipeline)
# ExperimentsPipeline.enqueue("translate", "Translate Test", TranslatePipeline)
# ExperimentsPipeline.enqueue("evaluate", "Evaluate Translations", EvaluationPipeline)

if __name__ == "__main__":
    config = Config(reader=WebNLGDataReader)

    res = ExperimentsPipeline.mutate({"config": config}) \
        .execute("WebNLG Experiments", cache_name="WebNLG_Exp")
