import os
import re
import unicodedata

from utils.file_system import save_temp

base = os.path.dirname(os.path.abspath(__file__))

# This should make the code work for  Python 2 / Python 3
try:
    unicode
except:
    unicode = str


def naive_tokenizer(s):
    s = re.sub(r'[^\x00-\x7F]', '', unicodedata.normalize("NFD", s))
    return ' '.join(re.split('(\W)', s.lower())).split()


def BLEU(hyps, refs, single_ref=False, tokenizer=None, hyp_tokenizer=None, ref_tokenizer=None, remove_empty=False):
    """
    hyps - array of strings
    refs - array of arrays containing strings. each array correlates to a single hypothesis
    """

    if len(hyps) == 0:
        return [0, 0, 0, 0, 0]

    # Add execution permissions
    os.popen("chmod +x " + base + "/multi-bleu.perl").read()

    if single_ref:
        refs = [[r] for r in refs]

    if remove_empty:
        refs = [ref for i, ref in enumerate(refs) if hyps[i] != ""]
        hyps = [hyp for hyp in hyps if hyp != ""]

    # Apply default tokenizer
    if not hyp_tokenizer and tokenizer:
        hyp_tokenizer = tokenizer
    if not ref_tokenizer and tokenizer:
        ref_tokenizer = tokenizer

    if hyp_tokenizer:
        hyps = [" ".join(t) for t in map(hyp_tokenizer, hyps)]

    if ref_tokenizer:
        refs = [[" ".join(t) for t in map(ref_tokenizer, ref)] for ref in refs]

    max_refs = max([len(ref) for ref in refs])
    refs = [ref + [""] * (max_refs - len(ref)) for ref in refs]

    dist_refs = list(zip(*refs))

    ref_path = []
    for i, refs in enumerate(dist_refs):
        ref_path.append(save_temp(list(map(unicode.lower, map(unicode, refs)))))

    hyps = list(map(unicode.lower, map(unicode, hyps)))
    hyp_path = save_temp(hyps)

    if all(map(lambda s: s == "", hyps)):
        return [0, 0, 0, 0, 0]

    cmd = base + "/multi-bleu.perl " + " ".join(ref_path) + " < " + hyp_path
    print(cmd)
    res = os.popen(cmd).read()
    # print res
    search = re.search(" (\d*[\.\d]*?), (\d*[\.\d]*?)\/(\d*[\.\d]*?)\/(\d*[\.\d]*?)\/(\d*[\.\d]*?) ", str(res))
    if search:
        scores = list(map(lambda k: float(k), search.groups()))
        return scores

    print(cmd)
    print(search)
    raise Exception(res)


if __name__ == "__main__":
    sen = "A small, TINY sentence with these_underlines to be tokenized!"
    print(naive_tokenizer(sen))
    print(BLEU([sen], [[sen]], tokenizer=naive_tokenizer))
