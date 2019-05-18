from collections import defaultdict
from utils.pipeline import Pipeline


def unique_plans_outputs(reader):
    plan_hyp_refs = defaultdict(lambda: ["", []])
    for d in reader.data:
        plan_hyp_refs[d.plan][0] = d.hyp
        plan_hyp_refs[d.plan][1].append(d.text)

    return dict(plan_hyp_refs)


def plans_output_single_file(plan_hyp_refs):
    return ["\n".join([plan, hyp, "---"] + refs) for plan, (hyp, refs) in plan_hyp_refs.items()]


TranslatePipeline = Pipeline()

TranslatePipeline.enqueue("translate", "Translate all plans",
                          lambda f, x: x["test-corpus"].copy().translate_plans(x["train-model"], x["train-planner"]))
TranslatePipeline.enqueue("post-process", "Post-process translated sentences",
                          lambda f, x: f["translate"].copy().post_process(x["train-reg"]))
TranslatePipeline.enqueue("plans-out", "Create a dictionary of outputs",
                          lambda f, x: unique_plans_outputs(f["post-process"]))
TranslatePipeline.enqueue("review", "Create hypothesis-references review file",
                          lambda f, x: "\n\n".join(["\n".join([plan, hyp, "---"] + refs)
                                                    for plan, (hyp, refs) in f["plans-out"].items()]), ext="txt")
TranslatePipeline.enqueue("hypothesis", "Create hypothesis file",
                          lambda f, x: "\n".join([hyp for plan, (hyp, refs) in f["plans-out"].items()]), ext="txt")
TranslatePipeline.enqueue("references", "Create references file",
                          lambda f, x: "\n\n".join(["\n".join(refs)
                                                    for plan, (hyp, refs) in f["plans-out"].items()]), ext="txt")
TranslatePipeline.enqueue("out", "Expose output for parent", lambda f, _: f["post-process"].copy())
