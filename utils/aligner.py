from functools import lru_cache

from utils.star import star

SENTENCE_BREAK = " _SENTENCE_BREAK_ "


def substring_indexes(substring, string):
    # Note, using regex is very slow here.
    last_found = -1  # Begin at -1 so the next position to search from is 0
    while True:
        # Find next index of substring, by starting after its last known position
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:
            break  # All occurrences have been found
        yield last_found


@lru_cache(maxsize=None)
def entities_order(sentence, entities):
    entities += tuple([SENTENCE_BREAK])
    ents_i = [(e, i) for e in entities for i in substring_indexes(e, sentence)]
    return list(sorted(ents_i, key=star(lambda e, i: i)))


def comp_order(ref, plan, skippable=set()):
    # Some pruning
    if len(plan) < len(ref):
        return False

    # New set reference
    skippable = set(skippable)

    for i in range(len(ref)):
        if ref[i] != plan[i]:
            if plan[i] in skippable:
                return comp_order(ref[i:], plan[i + 1:], skippable)
            else:
                return False
        if ref[i] != SENTENCE_BREAK:
            skippable.add(ref[i])

    return True
