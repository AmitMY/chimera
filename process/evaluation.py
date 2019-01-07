from utils.pipeline import Pipeline

EvaluationPipeline = Pipeline()

EvaluationPipeline.enqueue("evaluate", "Evaluate test reader", lambda f, x: x["translate"].evaluate())
EvaluationPipeline.enqueue("out", "Expose output for parent", lambda f, _: f["evaluate"].copy())
