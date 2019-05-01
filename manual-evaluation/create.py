import csv
from functools import lru_cache
from json import dump
from random import randint

import unicodedata

import re

from data.WebNLG.reader import WebNLG
from eval.bleu.eval import naive_tokenizer

corpus = WebNLG("../../data/WebNLG/test")

files = ["BIU_Chimera", "BIU_nmt"]

seen_limit = 970
ids = list(map(int, open("../../data/human/sample-ids.txt").read().strip().splitlines()))


@lru_cache()
def load_delex(f):
    return [' '.join(naive_tokenizer(s)).replace("  ", " ").replace("  ", " ") for s in
            open("../../data/submissions/delex/" + f + ".txt").read().lower().splitlines()]


if __name__ == "__main__":
    # samples = set()
    #
    # for f in files:
    #     s = load_delex(f)
    #
    #     for id in ids:
    #         if id <= seen_limit:
    #             id = id - 1
    #             sen = s[id].lower().replace("ent_", "<b>ent_").replace("_ent", "_ent</b>")
    #
    #             samples.add(
    #                 (id, relex(sen), tuple(map(lambda a: tuple(a.split(" | ") + [None]), corpus.graphs[id].rdf()))))
    #
    # samples = [{"id": id, "sen": sen, "rdf": rdf, "hal": 0} for id, sen, rdf in samples]
    # print(len(samples))
    # dump(samples, open("samples.json", "w"))
    #
    # print(samples[0])
