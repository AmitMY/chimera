import re
from itertools import chain
from os import listdir, path
from os.path import isdir

import xmltodict
from tqdm import tqdm

from data.WebNLG.rephrasing import rephrase, rephrase_if_must
from data.reader import DataReader, DataSetType, Datum
from utils.dbpedia import get_dbpedia_entity, pronouns
from utils.relex import RepresentsInt

FOR_MANUAL_EVAL = {18, 27, 37, 40, 41, 42, 55, 66, 69, 87, 90, 97, 101, 119, 130, 131, 133, 135, 142, 143, 144, 149,
                   150, 155, 169, 184, 188, 202, 209, 213, 223, 224, 225, 235, 239, 243, 257, 262, 274, 294, 301, 305,
                   310, 311, 325, 330, 334, 337, 355, 356, 359, 362, 366, 369, 373, 374, 375, 376, 383, 396, 400, 402,
                   403, 414, 419, 424, 438, 449, 451, 471, 472, 476, 483, 502, 511, 513, 518, 525, 536, 537, 538, 561,
                   569, 578, 581, 584, 585, 586, 591, 593, 602, 603, 619, 621, 623, 624, 632, 633, 648, 666, 669, 672,
                   691, 695, 696, 700, 701, 702, 706, 707, 717, 724, 729, 730, 734, 737, 740, 762, 768, 782, 786, 788,
                   793, 797, 805, 820, 825, 826, 827, 828, 833, 835, 836, 837, 840, 842, 857, 869, 871, 873, 876, 881,
                   889, 891, 899, 908, 913, 916, 993, 1010, 1020, 1038, 1042, 1075, 1080, 1091, 1107, 1131, 1139, 1173,
                   1175, 1181, 1183, 1205, 1208, 1224, 1261, 1265, 1276, 1288, 1298, 1309, 1325, 1329, 1341, 1345, 1363,
                   1368, 1393, 1399, 1405, 1436, 1440, 1445, 1463, 1465, 1466, 1504, 1505, 1523, 1530, 1537, 1542, 1577,
                   1579, 1582, 1606, 1613, 1614, 1620, 1639, 1648, 1668, 1673, 1692, 1704, 1721, 1733, 1752, 1755, 1763,
                   1772, 1774, 1776, 1782, 1784, 1794, 1796, 1807, 1810, 1852, 1859, 1861}


class RDFFileReader:
    def __init__(self, file_name):
        self.data = []

        content = open(file_name, encoding="utf-8").read()

        is_test_file = file_name.split("/")[-1] == "testdata_with_lex.xml"

        structure = xmltodict.parse(content)
        for i, entry in enumerate(structure["benchmark"]["entries"]["entry"]):
            triplets = [tuple(map(str.strip, r.split("|"))) for r in
                        self.triplets_from_object(entry["modifiedtripleset"], "mtriple")]
            sentences = list(self.extract_sentences(entry["lex"]))

            for s in sentences:
                info = {
                    "id": i,
                    "seen": not is_test_file or i <= 970,
                    "manual": is_test_file and i + 1 in FOR_MANUAL_EVAL and i <= 970
                }
                self.data.append(Datum(rdfs=triplets, text=s, info=info))

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

    def describe_entities(self):
        ents = set(chain.from_iterable([d.graph.nodes for d in self.data]))
        regnumber = re.compile(r'^\d+(\.\d*)?$')

        for ent in tqdm(ents):
            if ent[0] == '"':
                # print("Skipping", ent, "literal")
                pass
            elif ent[0] == '<':
                # print("Skipping", ent, "link")
                pass
            elif regnumber.match(ent):
                # print("Skipping", ent, "number")
                pass
            elif " " in ent:
                # print("Skipping", ent, "contains space")
                pass
            else:
                ps = pronouns(ent)
                if len(ps) > 0:
                    self.entities[ent.upper()] = ps

        return self


if __name__ == "__main__":
    reader = WebNLGDataReader(DataSetType.TEST)
    print(reader.data[-1])
    reader.generate_graphs()
    reader.describe_entities()
