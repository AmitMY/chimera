import re
from functools import lru_cache

from ordered_set import OrderedSet

from utils.delex import un_concat_entity

# Stringify
MONTHS = ['', 'January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']


@lru_cache(maxsize=None)
def get_entities(sentence):
    return OrderedSet(re.findall("ent_(.*?)_ent\\b", sentence, flags=re.IGNORECASE))


class Stringify:
    def __init__(self):
        pass

    @staticmethod
    def day_ordinal(n):
        return "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    @staticmethod
    def date(y, m, d):
        return MONTHS[int(m)] + " " + str(int(d)) + Stringify.day_ordinal(int(d)) + ", " + str(y)


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=None)
def relex(sentence: str):
    ents = {e: un_concat_entity(e) for e in get_entities(sentence)}

    for e, r in ents.items():
        # Dates
        splitted = r.split("-")
        if len(splitted) == 3 and all(map(RepresentsInt, splitted)):
            r = Stringify.date(splitted[0], splitted[1], splitted[2])

        # Numbers
        number_match = re.match("^(-?(\d+|\d{1,3}(,\d{3})*)(\.\d+)?)(\"(\((.*?)\)))?$", r)
        if number_match:
            groups = number_match.groups()
            number = float(groups[0])
            number = int(number) if number == int(number) else number
            unit = groups[6]
            if unit:
                r = str(number) + " " + unit
            else:
                r = str(number)

        insensitive_ent = re.compile(re.escape("ent_" + e + "_ent"), re.IGNORECASE)
        sentence = insensitive_ent.sub(r, sentence)


    return sentence


if __name__ == "__main__":
    print(relex("ent_13017_dot_0_quot__lp_minutes_rp__ent"))
    print(relex("ent_13017_dot_0_ent"))
    sen = "ENT_ABILENE_REGIONAL_AIRPORT_ENT serves the city of ENT_ABILENE_COMMA__TEXAS_ENT ."
    print(get_entities(sen))
    print(relex(sen))
