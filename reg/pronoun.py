import re
from collections import Counter
from itertools import chain
from typing import Dict, List

from reg.naive import NaiveREG
from reg.base import REG
from utils.delex import un_concat_entity


class PronounREG(NaiveREG):
    def generate(self, text: str, entities: Dict[str, List[str]]) -> str:
        new_text = []
        ent_c = Counter()
        for w in text.split():
            if self.is_ent(w):
                ent = un_concat_entity(w)
                ent_c[ent] += 1

                ent_underscore = ent.replace(" ", "_")
                if ent_c[ent] > 1 and ent_underscore in entities:
                    w = entities[ent_underscore][0]
                else:
                    w = self.process_word(ent, new_text[-1] if len(new_text) > 0 else None)

            new_text.append(w)

        return " ".join(new_text)
