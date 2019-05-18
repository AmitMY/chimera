import atexit
import re
from collections import Counter
from itertools import chain
from typing import Dict, List, Tuple

import torch

from data.WebNLG.reader import WebNLGDataReader
from data.reader import DataSetType
from eval.bleu.eval import naive_tokenizer
from reg.naive import NaiveREG
from tqdm import tqdm

from utils.dbpedia import all_pronouns
from utils.delex import un_concat_entity
from pytorch_pretrained_bert import BertForMaskedLM, tokenization

no_option = {"a", ",", ".", ";", ":", "!", "?"}


class BertREG(NaiveREG):
    def __init__(self, train_data=None, dev_data=None):
        super().__init__(train_data, dev_data)

        model_name = 'bert-base-uncased'
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.bert.to('cuda')
        self.tokenizer = tokenization.BertTokenizer.from_pretrained(model_name)
        self.bert.eval()

    def pred(self, tokens):
        tokens = ['[CLS]'] + self.tokenizer.tokenize(" ".join(tokens)) + ['[SEP]']
        # print(" ".join(tokens))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tens = torch.LongTensor(input_ids).unsqueeze(0).to('cuda')
        with torch.no_grad():
            res = self.bert(tens)[0, tokens.index("[MASK]")]
        probs, best_k = torch.topk(res, 100)
        best_k = [int(x) for x in best_k]
        best_k = self.tokenizer.convert_ids_to_tokens(best_k)
        return list(best_k)

    def generate(self, text: str, entities: Dict[str, List[str]]) -> Tuple[str, List]:
        ents_map = []

        new_text = []
        ent_c = Counter()
        tokens = text.split()

        for i, w in enumerate(tokens):
            if self.is_ent(w):
                ent = un_concat_entity(w)
                ent_c[ent] += 1

                ent_underscore = ent.replace(" ", "_").upper()

                if ent_c[ent] > 1:
                    pre = (" ".join(new_text).lower()).split(" ")
                    post = super().generate(" ".join(tokens[i + 1:]), entities)[0].lower().split()
                    middle = ["[MASK]", "("] + ent.lower().split() + [")"]
                    options = set(naive_tokenizer(ent)).difference(no_option)
                    if ent_underscore in entities:
                        options = options.union(set(entities[ent_underscore]))

                    # print()
                    # print(text)

                    pred = self.pred(pre + middle + post)

                    p_pred = pred[:10]

                    mapped_to = []

                    def get_index(w):
                        try:
                            return p_pred.index(w)
                        except:
                            return len(p_pred) + 1

                    # Check if "the" is most probable, of top 10
                    if "the" in p_pred and min([get_index(o) for o in options]) >= p_pred.index("the"):
                        new_text.append("the")
                        # print("BERT", "THE")

                        pred = self.pred(pre + ["the"] + middle + post)

                    if len(new_text) > 0 and new_text[-1].lower() == "the":
                        mapped_to.append("the")

                    ws = [p for p in pred if p in options]
                    # if len(ws) == 0 and len(ent.split()) > 1:
                    #     print("BERT failed...", pred)

                    w = ws[0] if len(ws) > 0 else ent
                    if w in all_pronouns and len(new_text) > 0 and new_text[-1].lower() == "the":
                        new_text.pop()
                        mapped_to.pop()

                    mapped_to.append(w)
                    ents_map.append((ent, " ".join(mapped_to)))

                    w = w.upper()
                    # print("BERT", w)
                else:
                    w = self.process_word(ent, new_text[-1] if len(new_text) > 0 else None)

            new_text.append(w)

        return " ".join(new_text), ents_map


if __name__ == "__main__":
    tests = [
        # HE
        "ENT_ALAN_BEAN_ENT is an UNITED STATES who was born in WHEELER, TEXAS . ENT_ALAN_BEAN_ENT is RETIRED .",
        # HIS
        "ENT_Adam_Holloway_ENT was born in Kent and his alma mater was Magdalene College, Cambridge. ENT_Adam_Holloway_ENT career began on 5 May 2005 and he fought in the Gulf war.",
        # The college
        "ENT_AWH_Engineering_College_ENT is located southeast of Mahe in Kuttikkattoor, Kerala, India. ENT_AWH_Engineering_College_ENT was established in 2001 and has 250 academic staff."
    ]

    data = WebNLGDataReader(DataSetType.TEST).generate_graphs().describe_entities()
    data.data = data.data[:len(tests)]
    for i, t in enumerate(tests):
        data.data[i].set_hyp(t)

    reg = BertREG()

    data.post_process(reg)

    for d in data.data:
        print(d.hyp)
    # for i in tqdm(range(100)):
    #     reg.pred(["hello", "[MASK]"])
