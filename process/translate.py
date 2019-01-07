from utils.pipeline import Pipeline

TranslatePipeline = Pipeline()
TranslatePipeline.enqueue("translate", "Translate all plans",
                          lambda f, x: x["test-corpus"].copy().translate_plans(x["train-model"]))
TranslatePipeline.enqueue("post-process", "Post-process translated sentences",
                          lambda f, _: f["translate"].copy().post_process())
TranslatePipeline.enqueue("out", "Expose output for parent", lambda f, _: f["post-process"].copy())
