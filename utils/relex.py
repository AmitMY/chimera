import re
from functools import lru_cache

from ordered_set import OrderedSet

from utils.delex import un_concat_entity

# Stringify
MONTHS = ['', 'January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']


@lru_cache(maxsize=None)
def get_entities(sentence):
    return re.findall("ent_(.*?)_ent\\b", sentence, flags=re.IGNORECASE)


class Stringify:
    def __init__(self):
        pass

    @staticmethod
    def day_ordinal(n):
        return "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    @staticmethod
    def date(y, m, d):
        return MONTHS[int(m)] + " " + str(int(d)) + Stringify.day_ordinal(int(d)) + ", " + str(y)

    @staticmethod
    def date_after_the(y, m, d):
        return str(int(d)) + Stringify.day_ordinal(int(d)) + " of " + MONTHS[int(m)] + ", " + str(y)


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
