import re
from itertools import chain
from typing import Dict, List, Tuple

from reg.base import REG
from utils.delex import un_concat_entity
from utils.relex import RepresentsInt, Stringify


class NaiveREG(REG):
    def __init__(self, train_data, dev_data):
        super().__init__(train_data, dev_data)

    def is_ent(self, word: str):
        ent_pattern = re.compile("^ent_(.*?)_ent$", flags=re.IGNORECASE)
        return ent_pattern.match(word)

    def process_word(self, w: str, prev_token: str):
        # Dates
        splitted = w.split("-")
        if len(splitted) == 3 and all(map(RepresentsInt, splitted)):
            if prev_token == "the":
                w = Stringify.date_after_the(splitted[0], splitted[1], splitted[2])
            else:
                w = Stringify.date(splitted[0], splitted[1], splitted[2])

        # Numbers
        number_match = re.match("^(-?(\d+|\d{1,3}(,\d{3})*)(\.\d+)?)(\"(\((.*?)\)))?$", w)
        if number_match:
            groups = number_match.groups()
            number = float(groups[0])
            number = int(number) if number == int(number) else number
            unit = groups[6]
            if unit:
                w = str(number) + " " + unit
            else:
                w = str(number)

        return w

    def generate(self, text: str, entities: Dict[str, List[str]]) -> Tuple[str, List]:
        new_text = []
        for w in text.split():
            if self.is_ent(w):
                w = self.process_word(un_concat_entity(w), new_text[-1] if len(new_text) > 0 else None)

            new_text.append(w)

        return " ".join(new_text), []


if __name__ == "__main__":
    print(un_concat_entity("ENT_AUDI_ENT"))
