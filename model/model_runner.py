from functools import lru_cache
from typing import List, Dict

from eval.bleu.eval import BLEU, naive_tokenizer


@lru_cache(maxsize=None)
def add_features(plan: str):
    # return plan
    paren = {"[", "]"}
    dir = {"<", ">"}
    words = plan.split(" ")
    features = ["S" if w in paren else ("D" if w in dir else ("E" if w[:4] == "ENT_" else "R")) for w in words]
    return " ".join([w + "\uFFE8" + f for w, f in zip(words, features)])  # Special Character!


def spread_translation_dict(for_translation):
    src_l = []
    tgt_l = []
    for src, tgts in for_translation.items():
        for tgt in tgts:
            src_l.append(src)
            tgt_l.append(tgt)

    return src_l, tgt_l


class Model:
    def translate(self, plans: List[str], opts=None) -> List[str]:  # Translate entire reader file using a model
        raise NotImplementedError("Must implement translate")

    def evaluate(self, ft: Dict[str, List[str]], opts=None):
        plans = list(ft.keys())
        references = list(ft.values())
        hypothesis = self.translate(plans, opts)

        return BLEU(hypothesis, references, tokenizer=naive_tokenizer)

    def evaluate_reader(self, reader):
        return self.evaluate(reader.for_translation())


class ModelRunner:
    def __init__(self, train_reader, dev_reader, features):
        self.train_ft = train_reader.for_translation()
        self.train_data = spread_translation_dict(self.train_ft)
        self.dev_ft = dev_reader.for_translation()
        self.dev_data = spread_translation_dict(self.dev_ft)

        self.features = features

    def expose_train(self):
        return "\n\n".join([p + "\n" + d for p, d in zip(*self.train_data)])

    def pre_process(self):  # Do any manipulations to the train and dev sets
        raise NotImplementedError("Must implement pre_process")

    def train(self, *args) -> str:  # Train your pre-processed files, save checkpoints, return checkpoints dir
        raise NotImplementedError("Must implement train")

    def find_best(self, checkpoints) -> Model:  # Return the model
        raise NotImplementedError("Must implement find_best")
