import json
from collections import Counter
from functools import lru_cache
from itertools import chain
from os.path import isfile
import requests

from utils.file_system import makedir

cache = "/tmp/dbpedia/"
makedir(cache)

DBPEDIA = "http://dbpedia.org/"


@lru_cache(maxsize=None)
def normalize_entity(entity: str):
    return entity \
        .replace("/", "%2F") \
        .replace("&", "%26") \
        .replace("+", "%2B")


@lru_cache(maxsize=None)
def get_dbpedia_entity(entity: str):
    entity = normalize_entity(entity)

    cache_ent = cache + entity + ".json"
    if isfile(cache_ent):
        f = open(cache_ent, "r")
        content = json.load(f)
        f.close()
        return content

    r = requests.get(url=DBPEDIA + 'data/' + entity + '.json')
    content = r.json()

    f = open(cache_ent, "w")
    json.dump(content, f)
    f.close()

    return content


def english_value(entries):
    filtered = [e["value"] for e in entries if e["lang"] == "en"]
    if len(filtered) > 0:
        return filtered[0]

    return None


gender_pronouns = {
    "male": ["he", "him", "his", "himself"],
    "female": ["she", "her", "hers", "herself"],
    "inanimate": ["it", "its", "itself"],
    "plural": ["they", "them", "theirs"]
}
all_pronouns = set(chain.from_iterable(gender_pronouns.values()))


def pronouns(entity: str):
    dbpedia = get_dbpedia_entity(entity)

    ent_uri = DBPEDIA + 'resource/' + entity
    gender_uri = "http://xmlns.com/foaf/0.1/gender"
    abstract_uri = "http://dbpedia.org/ontology/abstract"

    if ent_uri not in dbpedia:
        # print(dbpedia)
        # raise ValueError("No URI - " + entity)
        return []

    if gender_uri in dbpedia[ent_uri]:
        gender = english_value(dbpedia[ent_uri][gender_uri])
        if gender is not None:
            return gender_pronouns[gender]

    if abstract_uri in dbpedia[ent_uri]:
        abstract = english_value(dbpedia[ent_uri][abstract_uri])
        if abstract is not None:
            words = Counter(abstract.lower().split())
            gender_by_words = {g: sum([words[w] for w in g_words]) for g, g_words in gender_pronouns.items()}
            gender = max(gender_by_words, key=gender_by_words.get)
            if gender_by_words[gender] == 0:
                gender = "inanimate" # Default
            return gender_pronouns[gender]

    return []


if __name__ == "__main__":
    print("Start")
    for e in ['Jalisco', 'Diane_Duane', '23rd_Street_(Manhattan)']:
        print(len(get_dbpedia_entity(e)))
    print("end")

    print(pronouns("United_States"))
    print(pronouns("Buzz_Aldrin"))
    print(pronouns("Hillary_Clinton"))
    print(pronouns("Italy_national_football_team"))
