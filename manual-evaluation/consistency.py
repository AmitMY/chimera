from json import load, dump


samples = load(open("samples.json", "r"))
ref = load(open("EMNLP-neural.json"))

ref_dic = {r["sen"]: r for r in ref}

for i, s in enumerate(samples):
    if s["sen"] in ref_dic:
        print("FOUND", i)
        samples[i] = ref_dic[s["sen"]]

dump(samples, open("samples.json", "w"))
