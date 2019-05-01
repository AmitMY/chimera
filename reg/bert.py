import re
from collections import Counter
from itertools import chain
from typing import Dict, List

import torch

from naive import NaiveREG
from tqdm import tqdm

from utils.delex import un_concat_entity
from pytorch_pretrained_bert import BertForMaskedLM, tokenization


class BertREG(NaiveREG):
    def __init__(self, train_data=None, dev_data=None):
        super().__init__(train_data, dev_data)

        model_name = 'bert-base-uncased'
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.bert.to('cuda')
        self.tokenizer = tokenization.BertTokenizer.from_pretrained(model_name)
        self.bert.eval()
        print("BERT loaded")

    def pred(self, tokens):
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tens = torch.LongTensor(input_ids).unsqueeze(0).to('cuda')
        with torch.no_grad():
            res = self.bert(tens)[0, tokens.index("[MASK]")]
        probs, best_k = torch.topk(res, 100)
        best_k = [int(x) for x in best_k]
        best_k = self.tokenizer.convert_ids_to_tokens(best_k)
        return list(best_k)

    def generate(self, text: str, entities: Dict[str, List[str]]) -> str:
        new_text = []
        ent_c = Counter()
        tokens = text.split()
        for i, w in enumerate(tokens):
            if self.is_ent(w):
                ent = un_concat_entity(w)
                ent_c[ent] += 1

                ent_underscore = ent.replace(" ", "_")
                if ent_c[ent] > 1 and ent_underscore in self.data.entities:
                    pre = new_text
                    post = super().generate(" ".join(tokens[i + 1:])).split()
                    middle = ["[MASK]", "(", ent, ")"]
                    options = set(self.data.entities[ent_underscore] + ent.split())

                    pred = self.pred(pre + middle + post)
                    if pred[0] == "the":
                        new_text.append("the")
                        pred = self.pred(pre + ["the"] + middle + post)
                        w = pred[0]
                    else:
                        w = [p for p in pred if p in options][0]
                else:
                    w = self.process_word(ent, new_text[-1] if len(new_text) > 0 else None)

            new_text.append(w)

        return " ".join(new_text)


if __name__ == "__main__":
    reg = BertREG()
    for i in tqdm(range(100)):
        reg.pred(["hello", "[MASK]"])
