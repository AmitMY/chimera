from functools import lru_cache
import regex as re
from nltk import sent_tokenize, word_tokenize

ALPHA = chr(2)  # Start of text
OMEGA = chr(3)  # End of text
SPLITABLES = {ALPHA, OMEGA, " ", ".", ",", ":", "-", "'", "(", ")", "?", "!", "&", ";", '"'}


@lru_cache(maxsize=None)
def tokenize(s):
    return word_tokenize(s.replace("|", " | "))


@lru_cache(maxsize=None)
def tokenize_sentences(text: str):
    text = re.sub(r" no\. ent_(\d)", r" shorthand_number ent_\1", text, flags=re.IGNORECASE)
    return [s.replace("shorthand_number", "no.") for s in sent_tokenize(text)]


if __name__ == "__main__":
    print(tokenize_sentences(
        "ent_acharya_institute_of_technology_ent is affiliated with ent_visvesvaraya_technological_university_ent which is in ent_belgaum_ent. the institute itself is in ent_india_ent 's ent_karnataka_ent state and its full address is ent_in_soldevanahalli_comma__acharya_dr_dot__sarvapalli_radhakrishnan_road_comma__hessarghatta_main_road_comma__bangalore_–_560090_dot__ent. it was created in ent_2000_ent and its director is ent_dr_dot__g_dot__p_dot__prabhukumar_ent . they are no. ent_1_ent because of some reason"))
