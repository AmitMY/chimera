import multiprocessing
import pickle
import re
from collections import defaultdict
from enum import Enum
from multiprocessing.pool import Pool
from typing import List, Tuple, Dict, Callable

from tqdm import tqdm

from eval.bleu.eval import BLEU, naive_tokenizer
from model.model_runner import Model
from reg.reg import REG
from utils.aligner import entities_order, SENTENCE_BREAK, comp_order
from utils.delex import Delexicalize, concat_entity
from utils.graph import Graph
from utils.tokens import SPLITABLES, tokenize, tokenize_sentences


class DataSetType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Datum:
    def __init__(self, rdfs: List[Tuple[str, str, str]] = None,
                 graph: Graph = None,
                 title: str = None,
                 text: str = None,
                 delex: str = None,
                 hyp: str = None,
                 plan: str = None,
                 plans: List[str] = None):
        self.rdfs = rdfs
        self.graph = graph
        self.title = title
        self.text = text
        self.delex = delex
        self.hyp = hyp
        self.plan = plan
        self.plans = plans

    def set_graph(self, graph: Graph):
        self.graph = graph
        return self

    def set_text(self, text: str):
        self.text = text
        return self

    def set_delex(self, delex: str):
        self.delex = delex
        return self

    def set_hyp(self, hyp: str):
        self.hyp = hyp
        return self

    def set_plan(self, plan: str):
        self.plan = plan
        return self

    def set_plans(self, plans: str):
        self.plans = plans
        return self


def exhaustive_plan(g: Graph):
    return list(g.exhaustive_plan().linearizations())


def match_plan(d: Datum):
    g = d.graph
    s = d.delex

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
    DATASET = "GeneralDataReader"

    def __init__(self, data: List[Datum],
                 misspelling: Dict[str, str] = None,
                 rephrase: Tuple[Callable, Callable] = (None, None)):
        self.data = data
        self.misspelling = misspelling
        self.rephrase = rephrase

        self.entities = {}
        self.delex = Delexicalize(rephrase_f=self.rephrase[0], rephrase_if_must_f=self.rephrase[1])

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def generate_graphs(self):
        self.data = [d.set_graph(Graph(d.rdfs)) for d in self.data]
        return self

    def fix_spelling(self):
        if not self.misspelling:
            return self

        regex_splittable = "(\\" + "|\\".join(SPLITABLES) + ".)"

        for misspelling, fix in self.misspelling.items():
            source = regex_splittable + misspelling + regex_splittable
            target = "\1" + fix + "\2"

            self.data = [d.set_text(re.sub(source, target, d.text)) for d in self.data]

        return self

    def delex_single(self, text: str, ents: List[str], d: Datum):
        return self.delex.run(text, ents)

    def match_entities(self):
        self.data = [d if d.delex else d.set_delex(self.delex_single(d.text, d.graph.nodes, d)) for d in self.data]
        self.data = [d for d in self.data if d.delex]  # Filter out failed delex
        return self

    def match_plans(self):
        pool = Pool(multiprocessing.cpu_count() - 1)
        plans = list(reversed(list(tqdm(pool.imap(match_plan, list(reversed(self.data))), total=len(self.data)))))

        self.data = [d.set_plan(p) for d, plans in zip(self.data, plans) for p in plans]
        return self

    def exhaustive_plan(self):
        unique = {d.graph.unique_key(): d.graph for d in self.data}
        unique_graphs = list(reversed(list(unique.values())))

        # pool = Pool(multiprocessing.cpu_count() - 1)
        # plans = list(pool.imap(exhaustive_plan, unique_graphs))
        plans = [exhaustive_plan(p) for p in tqdm(unique_graphs)]
        graph_plans = {g.unique_key(): p for g, p in zip(unique_graphs, plans)}
        self.data = [d.set_plans(graph_plans[d.graph.unique_key()]) for d in self.data]
        return self

    def create_plans(self, planner):
        assert planner is not None
        unique = {d.graph.unique_key(): d.graph for d in self.data}
        unique_graphs = list(reversed(list(unique.values())))

        if planner.is_parallel:
            pool = Pool(multiprocessing.cpu_count() - 1)
            plans = list(tqdm(pool.imap(planner.plan_best, unique_graphs), total=len(unique_graphs)))
        else:
            plans = list(map(planner.plan_best, tqdm(unique_graphs)))
        graph_plan = {g.unique_key(): p for g, p in zip(unique_graphs, plans)}
        self.data = [d.set_plan(graph_plan[d.graph.unique_key()]) for d in self.data]
        return self

    def tokenize_plans(self):
        self.data = [d.set_plan(" ".join(tokenize(d.plan))) for d in self.data]
        return self

    def tokenize_delex(self):
        self.data = [d.set_delex(" ".join(tokenize(d.delex))) for d in self.data]
        return self

    def report(self):
        return "Length " + str(len(self.data))

    def for_translation(self):
        plan_sentences = defaultdict(list)
        for d in self.data:
            plan_sentences[d.plan].append(d.delex)

        return plan_sentences

    def get_plans(self):
        return list(set([d.plan for d in self.data]))

    def translate_plans(self, model: Model):
        plans = self.get_plans()
        translations = model.translate(plans)
        mapper = {p: t for p, t in zip(plans, translations)}
        self.data = [d.set_hyp(mapper[d.plan]) for d in self.data]
        return self

    def post_process(self, reg: REG):
        entities = self.entities if hasattr(self, 'entities') else {}

        def process(text: str):
            return reg.generate(text, entities)

        self.data = [d.set_hyp(process(d.hyp)) for d in self.data]
        return self

    def evaluate(self):
        plan_ref = defaultdict(list)
        plan_hyp = {}
        for d in self.data:
            plan_ref[d.plan].append(d.text)
            plan_hyp[d.plan] = d.hyp

        hypothesis = [plan_hyp[p] for p in plan_ref.keys()]
        references = list(plan_ref.values())

        return BLEU(hypothesis, references, tokenizer=naive_tokenizer)

    def describe_entities(self):
        return self
