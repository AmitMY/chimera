from functools import lru_cache
from typing import List

from eval.bleu.eval import BLEU, naive_tokenizer


@lru_cache(maxsize=None)
def add_features(plan: str):
    return plan
    # paren = {"[", "]"}
    # dir = {"<", ">"}
    # words = plan.split(" ")
    # features = ["S" if w in paren else ("D" if w in dir else ("E" if w[:4] == "ENT_" else "R")) for w in words]
    # return " ".join([w + "|" + f for w, f in zip(words, features)])


class Model:
    def translate(self, plans: List[str]) -> List[str]:  # Translate entire reader file using a model
        raise NotImplementedError("Must implement translate")

    def evaluate(self, reader):
        ft = reader.for_translation()
        plans = list(ft.keys())
        references = list(ft.values())
        hypothesis = self.translate(plans)

        return BLEU(hypothesis, references, tokenizer=naive_tokenizer)


class ModelRunner:
    def __init__(self, train_reader, dev_reader):
        self.train_reader = train_reader
        self.dev_reader = dev_reader

    def expose_train(self):
        return "\n\n".join([d.plan + "\n" + d.delex for d in self.train_reader.data])

    def pre_process(self):  # Do any manipulations to the train and dev sets
        raise NotImplementedError("Must implement pre_process")

    def train(self, *args) -> str:  # Train your pre-processed files, save checkpoints, return checkpoints dir
        raise NotImplementedError("Must implement train")

    def find_best(self, checkpoints) -> Model:  # Return the model
        raise NotImplementedError("Must implement find_best")
