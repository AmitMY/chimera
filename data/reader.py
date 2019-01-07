import multiprocessing
import numpy as np
import pickle
import re
from enum import Enum
from multiprocessing.pool import Pool
from typing import List, Tuple, Dict, Callable

from collections import defaultdict

from tqdm import tqdm

from eval.bleu.eval import BLEU, naive_tokenizer
from model.model_runner import Model
from scorer.scorer import ProductOfExperts
from utils.aligner import entities_order, SENTENCE_BREAK, comp_order
from utils.delex import Delexicalize, concat_entity
from utils.graph import Graph
from utils.relex import relex
from utils.tokens import SPLITABLES, tokenize, tokenize_sentences


class TextType(Enum):
    NO_ENTS = "NO_ENTITIES"
    ENTITIES = "ENTITIES"
    ENTITIES_AND_REFERENCES = "ENTITIES_AND_REFERENCES"


class DataSetType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


def create_plan(params: Tuple[Graph, ProductOfExperts]):
    g, scorer = params
    print(g.as_rdf())
    all_plans = g.exhaustive_plan().linearizations()
    if len(all_plans) == 0:
        return ""
    all_scores = [scorer.eval(p) for p in all_plans]
    max_i = np.argmax(all_scores)
    return all_plans[max_i]


def match_plan(params: Tuple[Graph, str]):
    g, s = params

    s_s = tokenize_sentences(s)

    components = []
    for sent in s_s:
        must_include = {n for n in g.nodes if concat_entity(n) in sent}
        must_exclude = g.nodes - must_include

        for n1, n2 in g.undirected_edges.keys():
            if n1 in must_include and n2 in must_exclude:
                must_exclude.remove(n2)

        components.append({
            "must_include": must_include,
            "must_exclude": must_exclude
        })

    nodes = tuple(map(concat_entity, g.nodes))

    possible_plans = g.constraint_graphs_plan(components).linearizations()
    final_plans = []

    for p in possible_plans:
        p_s = p.split(".")

        s_order = entities_order(SENTENCE_BREAK.join(s_s), nodes)
        p_order = entities_order(SENTENCE_BREAK.join(p_s), nodes)

        if comp_order([e for e, i in s_order], [e for e, i in p_order]):
            final_plans.append(p)

    return final_plans


class DataReader:
    def __init__(self, data: List[Tuple[List[Tuple[str, str, str]], str]],
                 text_type: TextType = TextType.NO_ENTS,
                 misspelling: Dict[str, str] = {},
                 rephrase: Tuple[Callable, Callable] = (None, None)):
        self.data = data
        self.text_type = text_type
        self.misspelling = misspelling
        self.rephrase = rephrase

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def generate_graphs(self):
        self.data = [(Graph(g), s) for g, s in self.data]
        return self

    def fix_spelling(self):
        regex_splittable = "(\\" + "|\\".join(SPLITABLES) + ".)"

        for misspelling, fix in self.misspelling.items():
            source = regex_splittable + misspelling + regex_splittable
            target = "\1" + fix + "\2"

            self.data = [(g, re.sub(source, target, s)) for g, s in self.data]

        return self

    def match_entities(self):
        if self.text_type != TextType.NO_ENTS:  # No need to match entities
            return self

        delex = Delexicalize(rephrase_f=self.rephrase[0], rephrase_if_must_f=self.rephrase[1])
        self.data = [(g, delex.run(s, g.nodes)) for g, s in self.data]
        self.data = [(g, s) for g, s in self.data if s]  # Filter out failed delex
        return self

    def match_plans(self):
        pool = Pool(multiprocessing.cpu_count() - 1)
        plans = list(reversed(list(pool.map(match_plan, list(reversed(self.data))))))
        self.data = [(g, p, s) for (g, s), plans in zip(self.data, plans) for p in plans]
        return self

    def create_plans(self, scorer):
        unique = {g.unique_key(): g for g, s in self.data}
        unique_graphs = list(unique.values())

        params = [(g, scorer) for g in reversed(unique_graphs)]
        pool = Pool(multiprocessing.cpu_count() - 1)
        plans = list(reversed(list(pool.imap(create_plan, params))))
        # plans = list(reversed(list([create_plan(p) for p in tqdm(params)])))
        graph_plans = {g.unique_key(): p for g, p in zip(unique_graphs, plans)}
        self.data = [(g, graph_plans[g.unique_key()], s) for g, s in self.data]
        return self

    def tokenize(self):
        self.data = [(g, " ".join(tokenize(p)), " ".join(tokenize(s))) for g, p, s in self.data]
        return self

    def report(self):
        return "Length " + str(len(self.data))

    def for_translation(self):
        plan_sentences = defaultdict(list)
        for g, p, s in self.data:
            plan_sentences[p].append(s)

        return plan_sentences

    def translate_plans(self, model: Model):
        plans = list(set([p for g, p, s in self.data]))
        translations = model.translate(plans)
        mapper = {p: t for p, t in zip(plans, translations)}
        self.data = [(g, p, mapper[p], s) for g, p, s in self.data]
        return self

    def post_process(self):
        def process(text: str):
            return relex(text)

        self.data = [(g, p, process(t), s) for g, p, t, s in self.data]
        return self

    def evaluate(self):
        plan_ref = defaultdict(list)
        plan_hyp = {}
        for g, p, t, s in self.data:
            plan_ref[p].append(s)
            plan_ref[p] = t

        hypothesis = [plan_hyp[p] for p in plan_ref.keys()]
        references = list(plan_ref.values())

        return BLEU(hypothesis, references, tokenizer=naive_tokenizer)
