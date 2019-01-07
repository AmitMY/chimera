from typing import List

from eval.bleu.eval import BLEU, naive_tokenizer


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

    def pre_process(self):  # Do any manipulations to the train and dev sets
        raise NotImplementedError("Must implement pre_process")

    def train(self, *args) -> str:  # Train your pre-processed files, save checkpoints, return checkpoints dir
        raise NotImplementedError("Must implement train")

    def find_best(self, checkpoints) -> Model:  # Return the model
        raise NotImplementedError("Must implement find_best")
