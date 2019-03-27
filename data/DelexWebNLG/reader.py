from itertools import chain
from os import listdir, path
from os.path import isdir

import xmltodict

from data.WebNLG.reader import misspelling
from data.WebNLG.rephrasing import rephrase, rephrase_if_must
from data.reader import DataReader, DataSetType, Datum
from utils.delex import concat_entity


class DelexRDFFileReader:
    def __init__(self, file_name):
        self.data = []

        content = open(file_name, encoding="utf-8").read()

        structure = xmltodict.parse(content)
        for entry in structure["benchmark"]["entries"]["entry"]:
            triplets = [tuple(map(str.strip, r.split("|"))) for r in
                        self.triplets_from_object(entry["modifiedtripleset"], "mtriple")]
            sentences = list(self.extract_sentences(entry["lex"]))

            for text, delex in sentences:
                self.data.append(Datum(rdfs=triplets, text=text, delex=delex))

    def extract_sentences(self, lex):
        sentences = lex
        if not isinstance(sentences, list):
            sentences = [sentences]

        for s in sentences:
            text = s["text"]
            template = s["template"]
            if not s["references"]:
                continue

            references = s["references"]["reference"]
            if not isinstance(references, list):
                references = [references]

            references = {r["@tag"]: r["@entity"] for r in references}
            for tag, ent in references.items():
                template = template.replace(tag, concat_entity(ent))

            yield text, template

    def triples_fix(self, triplets):
        if not isinstance(triplets, list):
            return [triplets]
        else:
            return map(lambda t: t, triplets)

    def triplets_from_object(self, obj, t_name):
        if not isinstance(obj, list):
            return self.triples_fix(obj[t_name])
        else:
            return [self.triples_fix(o[t_name]) for o in obj]


class DelexWebNLGDataReader(DataReader):
    DATASET = "DelexWebNLG"

    def __init__(self, set: DataSetType):
        files = self.recurse_files(path.join(path.dirname(path.realpath(__file__)), "raw", set.value))
        data = list(chain.from_iterable([DelexRDFFileReader(f).data for f in files]))

        super().__init__(data, misspelling=misspelling, rephrase=(rephrase, rephrase_if_must))

    def recurse_files(self, folder):
        if isdir(folder):
            return chain.from_iterable([self.recurse_files(folder + '/' + f) for f in listdir(folder)])
        return [folder]


if __name__ == "__main__":
    d = DelexWebNLGDataReader(DataSetType.TRAIN).data[1]
    print(d.text)
    print(d.delex)
    print(d.rdfs)
