import csv
from os import path

from data.reader import DataReader, DataSetType, Datum
from utils.delex import concat_entity
from utils.tokens import tokenize_sentences

relation_names = open(path.join(path.dirname(path.realpath(__file__)), "raw", "relations.vocab")).read().splitlines()


class AGENDADataReader(DataReader):
    DATASET = "AGENDA"

    def __init__(self, set: DataSetType):
        file_path = path.join(path.dirname(path.realpath(__file__)), "raw", "preRelease." + set.value + ".tsv")
        data = list(self.read_tsv(file_path))[:10]

        super().__init__(data)

    def read_tsv(self, file_path):
        rows = csv.reader(open(file_path, encoding="utf-8"), delimiter='\t')
        for title, entities, entity_types, relations, text in rows:
            entities = entities.split(" ; ")
            entity_types = [t.strip(">").strip("<") for t in entity_types.split(" ")]
            relations = [(entities[s], " ".join([entity_types[s], relation_names[r], entity_types[o]]), entities[o])
                         for s, r, o in map(lambda r: list(map(int, r.split(" "))), relations.split(" ; "))]

            relations += [("paper", "paper includes " + t, e) for e, t in zip(entities, entity_types)]

            for i, (ent, ent_type) in enumerate(zip(entities, entity_types)):
                text = text.replace("<" + ent_type + "_" + str(i) + ">", concat_entity(ent))

            text = text.replace("-lrb-", "(").replace("-rrb-", ")")

            yield Datum(text=text, delex=text, rdfs=relations, title=title)

    def tokenize_delex(self):  # Delex is already tokenized
        return self


if __name__ == "__main__":
    d = AGENDADataReader(DataSetType.TRAIN).data[1]

    print(d.title)
    print("\n")

    print("\n".join(tokenize_sentences(d.delex)))

    print("\n")
    for r in d.rdfs:
        print(r)
