from utils.pipeline import Pipeline

REGPipeline = Pipeline()
REGPipeline.enqueue("reg", "Learn planner",
                             lambda _, x: x["config"].reg(x["pre-process"]["train"], x["pre-process"]["dev"]))
REGPipeline.enqueue("out", "Expose the reg", lambda f, _: f["reg"])
