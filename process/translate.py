from utils.pipeline import Pipeline


def unique_plans_outputs(reader):
    mapper = {p: t for g, p, t, s in reader.data}
    print(len(mapper))
    return list(mapper.values())


TranslatePipeline = Pipeline()
TranslatePipeline.enqueue("translate", "Translate all plans",
                          lambda f, x: x["test-corpus"].copy().translate_plans(x["train-model"]))
TranslatePipeline.enqueue("post-process", "Post-process translated sentences",
                          lambda f, _: f["translate"].copy().post_process())
TranslatePipeline.enqueue("hypothesis", "Create hypothesis file",
                          lambda f, x: "\n".join(unique_plans_outputs(f["post-process"])))
TranslatePipeline.enqueue("out", "Expose output for parent", lambda f, _: f["post-process"].copy())
