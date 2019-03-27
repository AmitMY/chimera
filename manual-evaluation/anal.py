from json import load

from amt.manual.create import files, load_delex, ids, seen_limit, corpus
from utils.relex import relex

samples = load(open("samples.json"))

mapper = {(s["id"], s["sen"]): s for s in samples}

if __name__ == "__main__":
    for f in files:
        s = load_delex(f)

        scores = []

        for id in ids:
            if id <= seen_limit:
                id = id - 1
                sen = s[id].lower().replace("ent_", "<b>ent_").replace("_ent", "_ent</b>")

                key = (id, relex(sen))
                if key in mapper:
                    scores.append(mapper[(id, relex(sen))])

        rdfs = hal = exists = doesnt = wrong = 0
        for sc in scores:
            hal += sc["hal"]
            rdfs += len(sc["rdf"])
            exists += len([r for s, r, o, res in sc["rdf"] if res == "yes"])
            doesnt += len([r for s, r, o, res in sc["rdf"] if res == "no"])
            wrong += len([r for s, r, o, res in sc["rdf"] if res == "no-lex"])

        print(f, "rdfs", rdfs, "hallucinations", hal, "exists", exists, "doesn't", doesnt, "wrong-lex", wrong)
        print("verify", exists, "+", doesnt, "+", wrong, "=", exists+doesnt+wrong)
