from itertools import chain
from json import load

import numpy as np

if __name__ == "__main__":
    samples = load(open("samples.json"))
    rdfs = list(chain.from_iterable([s["rdf"] for s in samples]))
    hal = np.sum([s["hal"] for s in samples])

    total = len(rdfs)
    exists = len([r for s, r, o, res in rdfs if res == "yes"])
    doesnt = len([r for s, r, o, res in rdfs if res == "no"])
    wrong = len([r for s, r, o, res in rdfs if res == "no-lex"])
    wrong_reg = len([r for s, r, o, res in rdfs if res == "no-reg"])

    print([(s,r,o) for s, r, o, res in rdfs if res == "no"])

    print("rdfs", total, "hallucinations", hal, "exists", exists, "doesn't", doesnt, "wrong-lex", wrong, "wrong-reg", wrong_reg)
    print("verify", exists, "+", doesnt, "+", wrong, "+", wrong_reg, "=", total)
