from utils.pipeline import Pipeline

TrainPlannerPipeline = Pipeline()
TrainPlannerPipeline.enqueue("planner", "Learn planner",
                             lambda _, x: x["config"].planner.learn(x["pre-process"]["train"], x["pre-process"]["dev"]))
TrainPlannerPipeline.enqueue("out", "Expose the planner", lambda f, _: f["planner"])
