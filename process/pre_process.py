from data.WebNLG.reader import WebNLGDataReader
from data.reader import DataSetType
from utils.pipeline import Pipeline, ParallelPipeline

CorpusPreProcessPipeline = Pipeline()
CorpusPreProcessPipeline.enqueue("corpus", "Read Corpus", lambda f, x: x["config"].reader[f["set"]](f["set"]))
CorpusPreProcessPipeline.enqueue("graphify", "RDF to Graph", lambda f, _: f["corpus"].copy().generate_graphs())
CorpusPreProcessPipeline.enqueue("spelling", "Fix Spelling", lambda f, _: f["graphify"].copy().fix_spelling())

TestCorpusPreProcessPipeline = CorpusPreProcessPipeline.mutate({})  # Test does not need matching entities or plans

CorpusPreProcessPipeline.enqueue("match-ents", "Match Entities", lambda f, _: f["spelling"].copy().match_entities())
CorpusPreProcessPipeline.enqueue("match-plans", "Match Plans", lambda f, _: f["match-ents"].copy().match_plans())
CorpusPreProcessPipeline.enqueue("tokenize", "Tokenize Plans & Sentences",
                                 lambda f, _: f["match-plans"].copy().tokenize_plans().tokenize_delex())
CorpusPreProcessPipeline.enqueue("out", "Make output for parent", lambda f, _: f["tokenize"].copy())

TrainingPreProcessPipeline = ParallelPipeline()
TrainingPreProcessPipeline.enqueue("train", "Training Set", CorpusPreProcessPipeline.mutate({"set": DataSetType.TRAIN}))
TrainingPreProcessPipeline.enqueue("dev", "Dev Set", CorpusPreProcessPipeline.mutate({"set": DataSetType.DEV}))

TestCorpusPreProcessPipeline.enqueue("plan", "Generate best plan",
                                     lambda f, x: f["spelling"].copy().create_plans(x["learn-score"]))
TestCorpusPreProcessPipeline.enqueue("tokenize", "Tokenize Plans", lambda f, _: f["plan"].copy().tokenize_plans())
TestCorpusPreProcessPipeline.enqueue("out", "Make output for parent", lambda f, _: f["tokenize"].copy())

TestingPreProcessPipeline = Pipeline()
TestingPreProcessPipeline.enqueue("test", "Test Set", TestCorpusPreProcessPipeline.mutate({"set": DataSetType.TEST}))
TestingPreProcessPipeline.enqueue("out", "Make output for parent", lambda f, _: f["test"].copy())

if __name__ == "__main__":
    TrainingPreProcessPipeline.execute("Pre-Process")
    TestingPreProcessPipeline.execute("Pre-Process")
