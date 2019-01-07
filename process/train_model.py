from model.open_nmt import OpenNMTModelRunner
from utils.pipeline import Pipeline

train_opts = {
    "train_steps": 40000,
    "save_checkpoint_steps": 1000,
    "batch_size": 16,
    "word_vec_size": 300,
    "layers": 3,
    "copy_attn": None,
    "position_encoding": None
}

TrainModelPipeline = Pipeline()
TrainModelPipeline.enqueue("model", "Initialize OpenNMT",
                           lambda f, x: OpenNMTModelRunner(x["pre-process"]["train"], x["pre-process"]["dev"]))
TrainModelPipeline.enqueue("pre-process", "Pre-process Train and Dev", lambda f, x: f["model"].pre_process())
TrainModelPipeline.enqueue("train", "Train model", lambda f, x: f["model"].train(f["pre-process"], train_opts))
TrainModelPipeline.enqueue("find-best", "Find best model", lambda f, x: f["model"].find_best(f["train"]))
TrainModelPipeline.enqueue("out", "Output a model instance", lambda f, x: f["find-best"])
