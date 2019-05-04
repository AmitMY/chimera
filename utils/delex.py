import re
from functools import lru_cache

import dateparser
from Levenshtein import ratio

from data.WebNLG.rephrasing import rephrase, rephrase_if_must
from utils.star import star
from utils.tokens import SPLITABLES, ALPHA, OMEGA

dateRe = "^(\d\d\d\d)-(\d\d)-(\d\d)$"

placeholder_str = "_________"


def get_substrings(s):
    for i in range(0, len(s)):
        if s[i] in SPLITABLES:
            for j in range(i + 1, len(s)):
                if s[j] in SPLITABLES:
                    sub = s[i:j + 1]
                    if placeholder_str not in sub:
                        yield sub


def sorted_substrings(s):
    return sorted(list(get_substrings(s)), key=len)


def clean_extra(phrase):
    return phrase[1:len(phrase) - 1]


@lru_cache(maxsize=None)
def lev_ratio(s1, s2):
    return ratio(s1, s2)


def placeholder(i):
    # super unique, can't match it
    return placeholder_str + str(i) + placeholder_str


def clean_entity(e):
    return e.replace("_", " ")


def token_entity(e):
    return "|" + clean_entity(e) + "|"


@lru_cache(maxsize=None)
def concat_entity(e):
    e = str(e).strip().strip('"') \
        .replace(' ', '_') \
        .replace('(', '_LP_') \
        .replace(')', '_RP_') \
        .replace('.', '_DOT_') \
        .replace(',', '_COMMA_') \
        .replace("'", '_APOS_') \
        .replace('-', '_DASH_') \
        .replace(":", '_COLON_') \
        .replace('"', '_QUOT_') \
        .replace('&', '_AMP_') \
        .replace(';', '_SEMI_') \
        .replace('!', '_EXC_') \
        .replace('?', '_QUE_') \
        .replace('>', '_RT_') \
        .replace('<', '_LT_') \
        .replace('/', '_SLASH_')

    return "ENT_" + e.upper() + "_ENT"


@lru_cache(maxsize=None)
def un_concat_entity(e):
    if e[:4].lower() == "ent_":
        e = e[4:-4]

    return e.upper() \
        .replace('_LP_', '(') \
        .replace('_RP_', ')') \
        .replace('_DOT_', '.') \
        .replace('_COMMA_', ',') \
        .replace("_APOS_", "'") \
        .replace('_DASH_', '-') \
        .replace("_COLON_", ':') \
        .replace('_QUOT_', '"') \
        .replace('_AMP_', '&') \
        .replace('_SEMI_', ';') \
        .replace('_EXC_', '!') \
        .replace('_QUE_', '?') \
        .replace('_RT_', '>') \
        .replace('_LT_', '<') \
        .replace('_SLASH_', '/') \
        .replace('_', ' ')


class Delexicalize:
    def __init__(self, rephrase_f, rephrase_if_must_f, bound=0.75):
        self.rephrase = rephrase_f if rephrase_f else (lambda s: [s])
        self.rephrase_if_must = rephrase_if_must_f if rephrase_if_must_f else (lambda s: [s])
        self.bound = bound
        self.cutoff = self.bound * self.bound

    def closest_substring(self, phrases, subs):
        ratios = []

        for sub in subs:
            clean = clean_extra(sub)
            if clean != "":
                ratios.append((sub, max(
                    [0] + [max(lev_ratio(phrase, clean), lev_ratio(phrase, clean.replace(" ", ""))) for phrase in
                           phrases
                           if len(clean) * self.cutoff <= len(phrase) <= len(clean) / self.cutoff])))

        if len(ratios) == 0:
            return "", 0

        best_ratio = max(ratios, key=star(lambda s, r: r))[1]
        best_subs = [s for s, r in ratios if r == best_ratio]
        best_subs_by_length = sorted(best_subs, key=len, reverse=True)

        return best_subs_by_length, best_ratio

    def run(self, s, entities, concat_entities=True, allow_failed=False):
        # Look for entities by reverse length
        entities = list(entities)
        entities.sort(key=len, reverse=True)

        text_copy = str(s)

        s = ALPHA + s.lower() + OMEGA

        success = True
        for i, entity in enumerate(entities):
            if not success and not allow_failed:
                break

            subs = sorted_substrings(s)

            lower = entity.lower().replace("_", " ").strip('"')

            # If entity is date
            if re.match(dateRe, lower):
                [year, month, day] = map(int, lower.split('-'))
                rep = False
                for sub in subs:
                    try:
                        date = dateparser.parse(sub, languages=["en"])
                        if date and date.year == year and date.month == month and date.day == day:
                            s = s.replace(sub, sub[0] + lower + sub[-1])
                            rep = True
                            break
                    except:
                        pass
                if not rep:
                    success = False
                    continue

                # Update subs because changed the date
                subs = sorted_substrings(s)

            [n_subs, score] = self.closest_substring(self.rephrase(lower), subs)
            if score < self.bound:
                [n_subs, score] = self.closest_substring(self.rephrase_if_must(lower), subs)

                if score < self.bound:
                    # print("\nDelex Failed!")
                    # print(text_copy)
                    # print(clean_extra(s))
                    # print(entity)
                    # print("rephrase", self.rephrase(lower))
                    # print("rephrase", self.rephrase_if_must(lower))

                    success = False
                    continue

            for sub in n_subs:
                s = s.replace(sub, sub[0] + placeholder(i) + sub[-1])

        s = clean_extra(s)

        for i, entity in enumerate(entities):
            rep = concat_entity(entity) if concat_entities else token_entity(entity)
            s = s.replace(placeholder(i), rep)

        return s if success or allow_failed else False


if __name__ == "__main__":
    delex = Delexicalize(rephrase_f=rephrase, rephrase_if_must_f=rephrase_if_must)

    examples = [
        # [
        #     "Paris Culins and Gary Cohn are the creators of the comics character Bolt.",
        #     ['Gary_Cohn_(comics)', 'Paris_Cullins', 'Bolt_(comicsCharacter)']
        # ], [
        #     "William Anders was a member of the Apollo 8 crew (operated by NASA) and he retired on September 1st, 1969. Frank Borman was the commander of Apollo 8 and Buzz Aldrin was a back up pilot.",
        #     ['Frank_Borman', 'William_Anders', 'Buzz_Aldrin', 'Apollo_8', '"1969-09-01"', 'NASA']
        # ], [
        #     "The Accademia di Architettura di Mendrisio is located in Mendrisio, Switzerland. It was established in 1996 and its dean is Mario Botta. It has 600 students. The leader of Switzerland is Federal Chancellor Johann Schneider-Ammann.",
        #     ["Accademia_di_Architettura_di_Mendrisio", "Switzerland", "Mario_Botta", "1996", "600", "Mendrisio",
        #      "Federal_Chancellor_of_Switzerland", "Johann_Schneider-Ammann"]
        # ],

        [
            "ENT_ANDREWS_COUNTY_AIRPORT_ENT runway is 8 meters long.",
            ['andrews county airport', '8.0']
        ], [
            "5 is.",
            ['5']
        ],
        [
            "The architect of 200 Public Square is HOK.",
            ['200_Public_Square ', 'HOK_(firm)']
        ]
    ]

    for sentence, entities in examples:
        # print(sentence)the architect of
        delex.run(sentence, entities, True)
