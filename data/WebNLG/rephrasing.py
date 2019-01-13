import itertools
import re

rephrasing = {
    # Add an acronym database
    "united states": ["u.s.", "u.s.a.", "us", "usa", "america", "american"],
    "united kingdom": ["u.k.", "uk"],
    "united states air force": ["usaf", "u.s.a.f"],
    "new york": ["ny", "n.y."],
    "new jersey": ["nj", "n.j."],
    "f.c.": ["fc"],
    "submarine": ["sub"],
    "world war ii": ["ww ii", "second world war"],
    "world war i": ["ww i", "first world war"],

    "greece": ["greek"],
    "canada": ["canadian"],
    "italy": ["italian"],
    "america": ["american"],
    "india": ["indian"],
    "singing": ["sings"],
    "conservative party (uk)": ["tories"],
    "ethiopia": ["ethiopian"],
}

rephrasing_must = {
    # Add a rephrasing database
    " language": "",
    " music": "",
    "kingdom of ": "",
    "new york city": "new york",
    "secretary of state of vermont": "secretary of vermont"
}


def rephrase(entity):
    phrasings = {entity}

    for s, rephs in rephrasing.items():
        for p in filter(lambda p: s in p, set(phrasings)):
            for r in rephs:
                phrasings.add(p.replace(s, r))

    # Allow rephrase "a/b/.../z" -> every permutation
    for p in set(phrasings):
        for permutation in itertools.permutations(p.split("/")):
            phrasings.add("/".join(permutation))

    # Allow rephrase "number (unit)" -> "number unit", "number unit-short"
    for p in set(phrasings):
        match = re.match("^(-?(\d+|\d{1,3}(,\d{3})*)(\.\d+)?)( (\((.*?)\)))?$", p)
        if match:
            groups = match.groups()
            number = float(groups[0])
            unit = groups[6]

            number_phrasing = [
                str(number),
                str("{:,}".format(number))
            ]
            if round(number) == number:
                number_phrasing.append(str(round(number)))
                number_phrasing.append(str("{:,}".format(round(number))))

            if unit:
                couple = None
                words = [unit]

                if unit == "metres":
                    couple = "m"
                    words = [unit, "meters"]
                elif unit == "millimetres":
                    couple = "mm"
                elif unit == "centimetres":
                    couple = "cm"
                elif unit == "kilometres":
                    couple = "km"
                elif unit == "kilograms":
                    couple = "kg"
                elif unit == "litres":
                    couple = "l"
                elif unit == "inches":
                    couple = "''"
                elif unit in ["degreecelsius", "degreeklsius"]:
                    words = ["degrees celsius"]
                elif unit == "grampercubiccentimetres":
                    words = ["grams per cubic centimetre"]
                elif unit == "kilometreperseconds":
                    words = ["kilometres per second", "km/s", "km/sec", "km per second", "km per sec"]
                elif unit in ["squarekilometres", "square kilometres"]:
                    words = ["square kilometres", "sq km"]
                elif unit == "cubiccentimetres":
                    couple = "cc"
                    words = ["cubic centimetres"]
                elif unit in ["cubic inches", "days", "tonnes", "square metres",
                              "inhabitants per square kilometre", "kelvins"]:
                    pass
                else:
                    raise ValueError(unit + " is unknown")

                for np in number_phrasing:
                    if couple:
                        phrasings.add(np + " " + couple)
                        phrasings.add(np + couple)
                    for word in words:
                        phrasings.add(np + " " + word)
            else:
                for np in number_phrasing:
                    phrasings.add(np)

    # Allow rephrase "word1 (word2)" -> "word2 word1"
    for p in set(phrasings):
        match = re.match("^(.* ?) \((.* ?)\)$", p)
        if match:
            groups = match.groups()
            s = groups[0]
            m = groups[1]
            phrasings.add(s + " " + m)
            phrasings.add(m + " " + s)

    return set(phrasings)


def rephrase_if_must(entity):
    phrasings = {entity}

    for s, rephs in rephrasing_must.items():
        for p in filter(lambda p: s in p, set(phrasings)):
            for r in rephs:
                phrasings.add(p.replace(s, r))

    # Allow removing parenthesis "word1 (word2)" -> "word1"
    for p in set(phrasings):
        match = re.match("^(.* ?) \((.* ?)\)$", p)
        if match:
            groups = match.groups()
            phrasings.add(groups[0])

    # Allow rephrase "word1 (word2) word3?" -> "word1( word3)"
    for p in set(phrasings):
        match = re.match("^(.*?) \((.*?)\)( .*)?$", p)
        if match:
            groups = match.groups()
            s = groups[0]
            m = groups[2]
            phrasings.add(s + " " + m if m else "")

    # Allow rephrase "a b ... z" -> every permutation
    # for p in set(phrasings):
    #     for permutation in itertools.permutations(p.split(" ")):
    #         phrasings.add(" ".join(permutation))

    phrasings = set(phrasings)
    if "" in phrasings:
        phrasings.remove("")
    return phrasings
