import json

from data.WebNLG.reader import WebNLGDataReader
from data.reader import DataSetType
from utils.error_bar import error_bar
from utils.pipeline import Pipeline, ParallelPipeline

CorpusPreProcessPipeline = Pipeline()
CorpusPreProcessPipeline.enqueue("corpus", "Read Corpus", lambda f, x: x["config"].reader[f["set"]](f["set"]))
CorpusPreProcessPipeline.enqueue("graphify", "RDF to Graph", lambda f, _: f["corpus"].copy().generate_graphs())
CorpusPreProcessPipeline.enqueue("spelling", "Fix Spelling", lambda f, _: f["graphify"].copy().fix_spelling())
CorpusPreProcessPipeline.enqueue("entities", "Describe entities", lambda f, _: f["spelling"].copy().describe_entities())

TestCorpusPreProcessPipeline = CorpusPreProcessPipeline.mutate({})  # Test does not need matching entities or plans

CorpusPreProcessPipeline.enqueue("match-ents", "Match Entities", lambda f, _: f["entities"].copy().match_entities())
CorpusPreProcessPipeline.enqueue("match-plans", "Match Plans", lambda f, _: f["match-ents"].copy().match_plans())
CorpusPreProcessPipeline.enqueue("tokenize", "Tokenize Plans & Sentences",
                                 lambda f, _: f["match-plans"].copy().tokenize_plans().tokenize_delex())
CorpusPreProcessPipeline.enqueue("to-json", "Export in a readable format",
                                 lambda f, _: json.dumps(f["tokenize"].export()), ext="json")
CorpusPreProcessPipeline.enqueue("out", "Make output for parent", lambda f, _: f["tokenize"].copy())

TrainingPreProcessPipeline = ParallelPipeline()
TrainingPreProcessPipeline.enqueue("train", "Training Set", CorpusPreProcessPipeline.mutate({"set": DataSetType.TRAIN}))
TrainingPreProcessPipeline.enqueue("dev", "Dev Set", CorpusPreProcessPipeline.mutate({"set": DataSetType.DEV}))

TestCorpusPreProcessPipeline.enqueue("plan", "Generate best plan",
                                     lambda f, x: f["entities"].copy().create_plans(x["train-planner"]))
TestCorpusPreProcessPipeline.enqueue("timing", "Chart the timing",
                                     lambda f, x: error_bar(f["plan"].timing, "Time (seconds)", "#Edges"), ext="pdf")
TestCorpusPreProcessPipeline.enqueue("tokenize", "Tokenize Plans", lambda f, _: f["plan"].copy().tokenize_plans())
TestCorpusPreProcessPipeline.enqueue("out", "Make output for parent", lambda f, _: f["tokenize"].copy())

TestingPreProcessPipeline = Pipeline()
TestingPreProcessPipeline.enqueue("test", "Test Set", TestCorpusPreProcessPipeline.mutate({"set": DataSetType.TEST}))
TestingPreProcessPipeline.enqueue("out", "Make output for parent", lambda f, _: f["test"].copy())

if __name__ == "__main__":
    TrainingPreProcessPipeline.execute("Pre-Process")
    TestingPreProcessPipeline.execute("Pre-Process")
