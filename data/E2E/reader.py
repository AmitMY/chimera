import csv
from collections import defaultdict
from itertools import chain
from os import listdir, path
from os.path import isdir
from typing import List

import xmltodict

from data.E2E.rephrasing import rephrase, rephrase_if_must
from data.reader import DataReader, DataSetType, Datum
from utils.delex import concat_entity
from utils.tokens import tokenize

DONT_DELEX = {"priceRange", "familyFriendly", "customer rating"}


class E2EDataReader(DataReader):
    DATASET = "E2E"

    def __init__(self, set: DataSetType):
        reader = csv.reader(open(path.join(path.dirname(path.realpath(__file__)), "raw", set.value + ".csv"), "r"))
        rows = list(reader)[1:]

        data = [E2EDataReader.parse_csv_row(*r) for r in rows]

        super().__init__(data, rephrase=(rephrase, rephrase_if_must))

    @staticmethod
    def parse_csv_row(info, text):
        obj = {}
        for block in map(str.strip, info.split(",")):
            r, o = block[:-1].split("[")
            if r == "familyFriendly":
                if o not in ["yes", "no"]:
                    raise ValueError("Unknown familyFriendly value: " + o)
                o = "family-friendly" if o == "yes" else "not family-friendly"
            obj[r] = o

        if "name" not in obj:
            print(info)
            print(text)
            raise ValueError("Data must contain name")

        rdfs = []
        for key, value in obj.items():
            if key != "name":
                rdfs.append((obj["name"], key, value))

        return Datum(rdfs=rdfs, text=text.replace("  ", " "))

    def delex_single(self, text: str, ents: List[str], d: Datum):
        delex = super().delex_single(text, ents, d)
        meta = self.get_meta(d)
        if delex:
            delex = delex \
                .replace(" it ", " " + concat_entity(meta["name"]) + " ") \
                .replace(" they ", " " + concat_entity(meta["name"]) + " ")
        return delex

    def get_meta(self, d: Datum):
        # Location name, and near can be replaced as constants, for unseen data.
        name = d.rdfs[0][0]
        near = [rdf[2] for rdf in d.rdfs if rdf[1] == "near"]
        return {
            "name": name,
            "near": None if len(near) == 0 else near[0]
        }

    def for_translation(self):
        plan_sentences = defaultdict(list)
        for d in self.data:
            meta = self.get_meta(d)
            plan = d.plan.replace(concat_entity(meta["name"]), concat_entity("name"))
            delex = d.delex
            if delex:
                ents = [meta["name"]] + [o for s, r, o in d.rdfs if r not in DONT_DELEX]
                delex = self.delex_single(d.text, ents, d)
                delex = delex.replace(concat_entity(meta["name"]), concat_entity("name"))
                delex = " ".join(tokenize(delex))

            if meta["near"]:
                near_val = concat_entity(meta["near"])
                near_placeholder = concat_entity("near")
                plan = plan.replace(near_val, near_placeholder)
                delex = delex.replace(near_val, near_placeholder)

            plan_sentences[plan].append(delex)

        return plan_sentences

    def post_process(self):
        def rename(d: Datum, text: str):
            meta = self.get_meta(d)
            # First occurrence full name, others "it"/"they"
            text = text \
                .replace(concat_entity("name"), concat_entity(meta["name"]), 1) \
                .replace(concat_entity("name") + " are", "THEY are") \
                .replace(concat_entity("name") + " have", "THEY have") \
                .replace(concat_entity("name"), "IT")

            if meta["near"]:
                text = text.replace(concat_entity("near"), concat_entity(meta["near"]))
            return text

        self.data = [d.set_hyp(rename(d, d.hyp)) for d in self.data]

        super().post_process()
        return self


if __name__ == "__main__":
    reader = E2EDataReader(DataSetType.DEV)
    reader.data = reader.data[:1000]
    reader.generate_graphs().match_entities().match_plans()

    # for datum in reader.data:
    #     print()
    #     print(datum.rdfs)
    #     print(datum.text)
    #     print(datum.delex)
    #     print(datum.plan)
