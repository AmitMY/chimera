import itertools
import re

rephrasing = {
    "1 out of 5": ["one star", "1 star", "poor reviews", "low rated"],
    "2 out of 5": ["two stars", "2 stars"],
    "3 out of 5": ["three stars", "3 stars", "decent"],
    "4 out of 5": ["four stars", "4 stars"],
    "5 out of 5": ["five stars", "5 stars", "of high quality", "highly-rated"],
    "not family-friendly": ["adult", "no good for families", "people or families aren't allowed",
                            "no kids", "not kid friendly", "not child friendly", "not allow families",
                            "not considered child friendly", "not suitable for children"],
    "family-friendly": ["kid friendly", "child friendly", "a place to bring the whole family", "children family",
                        "welcomes children", "caters to children", "opened to all age groups",
                        "a good place to bring children", "suitable for children"],
    "riverside": ["near the river", "on the river"],
    "cheap": ["inexpensive", "low priced"],
}


def rephrase(entity):
    phrasings = {entity}

    if entity in rephrasing:
        for r in rephrasing[entity]:
            phrasings.add(r)

    return phrasings


rephrasing_must = {
    "moderate": ["medium", "mid"],
    "coffee shop": ["restaurant", "coffee"],
    "family-friendly": ["family"],
    "1 out of 5": ["1"],
    "2 out of 5": ["2"],
    "3 out of 5": ["3"],
    "4 out of 5": ["4"],
    "5 out of 5": ["5", "good", "excellent"],
    "high": ["top", "popular"],
    "low": ["not good", "bad"],

    "£20-25": ["average", "20 to 25", "moderately"],
    "more than £30": ["rather expensive"],
    "less than £20": ["under £20", "low priced"]
}


def rephrase_if_must(entity):
    phrasings = {entity}

    if entity in rephrasing_must:
        for r in rephrasing_must[entity]:
            phrasings.add(r)

    return phrasings
