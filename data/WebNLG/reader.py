from itertools import chain
from os import listdir, path
from os.path import isdir

import xmltodict

from data.WebNLG.rephrasing import rephrase, rephrase_if_must
from data.reader import DataReader, DataSetType, Datum


class RDFFileReader:
    def __init__(self, file_name):
        self.data = []

        content = open(file_name, encoding="utf-8").read()

        structure = xmltodict.parse(content)
        for entry in structure["benchmark"]["entries"]["entry"]:
            triplets = [tuple(map(str.strip, r.split("|"))) for r in self.triplets_from_object(entry["modifiedtripleset"], "mtriple")]
            sentences = list(self.extract_sentences(entry["lex"]))

            for s in sentences:
                self.data.append(Datum(rdfs=triplets, text=s))

    def extract_sentences(self, lex):
        sentences = lex
        if not isinstance(sentences, list):
            sentences = [sentences]

        return map(lambda s: s["#text"], sentences)

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


misspelling = {
    "accademiz": "academia",
    "withreference": "with reference",
    "thememorial": "the memorial",
    "unreleated": "unrelated",
    "varation": "variation",
    "variatons": "variations",
    "youthclub": "youth club",
    "oprated": "operated",
    "originaly": "originally",
    "origintes": "originates",
    "poacea": "poaceae",
    "posgraduayed": "postgraduate",
    "prevously": "previously",
    "publshed": "published",
    "punlished": "published",
    "recor": "record",
    "relgiion": "religion",
    "runwiay": "runway",
    "sequeled": "runway",
    "sppoken": "spoken",
    "studiies": "studies",
    "sytle": "style",
    "tboh": "both",
    "whic": "which",
    "identfier": "identifier",
    "idenitifier": "identifier",
    "igredient": "ingredients",
    "ingridient": "ingredients",
    "inclusdes": "includes",
    "indain": "indian",
    "leaderr": "leader",
    "legue": "league",
    "lenght": "length",
    "loaction": "location",
    "locaated": "located",
    "locatedd": "located",
    "locationa": "location",
    "managerof": "manager of",
    "manhattern": "manhattan",
    "memberrs": "members",
    "menbers": "members",
    "meteres": "metres",
    "numbere": "number",
    "numberr": "number",
    "notablework": "notable work",
    "7and": "7 and",
    "abbreivated": "abbreviated",
    "abreviated": "abbreviated",
    "abreviation": "abbreviation",
    "addres": "address",
    "abbreviatedform": "abbreviated form",
    "aerbaijan": "azerbaijan",
    "azerbijan": "azerbaijan",
    "affilaited": "affiliated",
    "affliate": "affiliate",
    "aircfrafts": "aircraft",
    "aircrafts": "aircraft",
    "aircarft": "aircraft",
    "airpor": "airport",
    "in augurated": "inaugurated",
    "inagurated": "inaugurated",
    "inaugrated": "inaugurated",
    "ausitin": "austin",
    "coccer": "soccer",
    "comanded": "commanded",
    "constructionof": "construction of",
    "counrty": "country",
    "countyof": "county of",
    "creater": "creator",
    "currecncy": "currency",
    "denonym": "demonym",
    "discipine": "discipline",
    "engish": "english",
    "establishedin": "established in",
    "ethinic": "ethnic",
    "ethiopa": "ethiopia",
    "ethipoia": "ethiopia",
    "eceived": "received",
    "ffiliated": "affiliated",
    "fullname": "full name",
    "grop": "group"
}


class WebNLGDataReader(DataReader):
    DATASET = "WebNLG"

    def __init__(self, set: DataSetType):
        files = self.recurse_files(path.join(path.dirname(path.realpath(__file__)), "raw", set.value))
        data = list(chain.from_iterable([RDFFileReader(f).data for f in files]))

        super().__init__(data, misspelling=misspelling, rephrase=(rephrase, rephrase_if_must))

    def recurse_files(self, folder):
        if isdir(folder):
            return chain.from_iterable([self.recurse_files(folder + '/' + f) for f in listdir(folder)])
        return [folder]


if __name__ == "__main__":
    print(WebNLGDataReader(DataSetType.TRAIN).data[-1])
