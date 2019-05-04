from utils.pipeline import Pipeline

# CoverageEvaluationPipeline = Pipeline()
# CoverageEvaluationPipeline.enqueue("plan-all", "Plan all & score on test set",
#                                    lambda f, x: x["test-corpus"].copy().exhaustive_plan(x["train-planner"]))
# CoverageEvaluationPipeline.enqueue("print", "Print stuff",
#                                    lambda f, x: "\n".join([str(len(d.graph.edges)) + " - " + str(len(d.plans)) for d in f["plan-all"].data]), ext="txt")

EvaluationPipeline = Pipeline()
EvaluationPipeline.enqueue("bleu", "Evaluate test reader", lambda f, x: x["translate"].evaluate())
# EvaluationPipeline.enqueue("coverage", "Coverage evaluation", CoverageEvaluationPipeline)
