import multiprocessing
import pickle
import re
import zlib
from collections import defaultdict, Counter
from enum import Enum
from itertools import chain
from multiprocessing.pool import Pool
from typing import List, Tuple, Dict, Callable

import numpy as np
from tqdm import tqdm
import time

from eval.bleu.eval import BLEU, naive_tokenizer
from model.model_runner import Model
from reg.base import REG
from utils.aligner import entities_order, SENTENCE_BREAK, comp_order
from utils.delex import Delexicalize, concat_entity
from utils.graph import Graph
from utils.out_of import out_of
from utils.relex import get_entities
from utils.tokens import SPLITABLES, tokenize, tokenize_sentences


class DataSetType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Datum:
    def __init__(self, rdfs: List[Tuple[str, str, str]] = None,
                 graph: Graph = None,
                 info=None,
                 title: str = None,
                 text: str = None,
                 delex: str = None,
                 hyp: str = None,
                 plan: str = None,
                 plans: List[str] = None):
        self.rdfs = rdfs
        self.graph = graph
        self.info = info
        self.title = title
        self.text = text
        self.delex = delex
        self.hyp = hyp
        self.plan = plan
        self.plans = plans

        self.plan_changes = 0

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

        if not hasattr(self, "plan_changes"):  # TODO remove after EMNLP
            self.plan_changes = 1

        self.plan_changes += 1
        return self

    def set_plans(self, plans: List[str]):
        self.plans = plans
        return self


def exhaustive_plan(g: Graph, planner):
    plans = list(planner.plan_all(g))
    scores = planner.scores([(g, p) for p in plans])
    p_scores = {p: scores[i] for i, p in enumerate(plans)}
    return sorted(plans, key=lambda p: p_scores[p], reverse=True)


def compress_plans(plans: List[str]) -> str:
    return zlib.compress("\n".join(plans).encode("utf-8"))


def exhaustive_plan_compress(input):
    g, planner = input
    return compress_plans(exhaustive_plan(g, planner))


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

    if len(final_plans) == 0:
        return []
    return [max(final_plans, key=lambda p: len(list(filter(lambda w: w == ">", p.split()))))]


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

        self.timing = {}

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

    def exhaustive_plan(self, planner):
        unique = {d.graph.unique_key(): d.graph for d in self.data}
        unique_graphs = list(reversed(list(unique.values())))

        plan_iter = ((g, planner) for g in unique_graphs)

        # pool = Pool(multiprocessing.cpu_count() - 1)
        # plans = list(tqdm(pool.imap(exhaustive_plan_compress, plan_iter),
        #                   total=len(unique_graphs)))
        plans = [exhaustive_plan_compress(g_p) for g_p in tqdm(list(plan_iter))]

        graph_plans = {g.unique_key(): p for g, p in zip(unique_graphs, plans)}
        self.data = [d.set_plans(graph_plans[d.graph.unique_key()]) for d in self.data]
        return self

    def create_plans(self, planner):
        assert planner is not None

        unique = {d.graph.unique_key(): d.graph for d in self.data}
        unique_graphs = list(reversed(list(unique.values())))

        # if planner.is_parallel:
        #     pool = Pool(multiprocessing.cpu_count() - 1)
        #     plans = list(tqdm(pool.imap(planner.plan_best, unique_graphs), total=len(unique_graphs)))
        # else:
        plans = []
        for g in tqdm(unique_graphs):
            start = time.time()
            plans.append(planner.plan_best(g))
            g_size = len(g.edges)
            if g_size not in self.timing:
                self.timing[g_size] = []
            self.timing[g_size].append(time.time() - start)

        graph_plan = {g.unique_key(): p for g, p in zip(unique_graphs, plans)}
        for d in self.data:
            plans = graph_plan[d.graph.unique_key()]
            if isinstance(plans, list):
                d.set_plan(plans[0])
                d.set_plans(plans[1:])
            else:
                d.set_plan(plans)
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

    def translate_plans(self, model: Model, planner, opts=None):
        data = self.data

        fallback = {}

        for _ in range(50):
            plans = [d.plan for d in data]

            translations = model.translate(plans, opts)

            for d, p, t in zip(data, plans, translations):
                is_covered_ent, is_covered_order = self.single_coverage(p, t)
                if is_covered_order:
                    d.set_hyp(t)

                graph_key = d.graph.unique_key()
                if not (graph_key in fallback) or (is_covered_ent and not fallback[graph_key][2]):
                    fallback[graph_key] = (p, t, is_covered_ent)

            data = list(filter(lambda d: d.hyp is None, data))
            if len(data) == 0:
                break

            if planner.re_plan == "PREMADE":
                for d in data:
                    plans = d.plans
                    if len(plans) > 0:
                        d.set_plans(plans[1:])
                        d.set_plan(plans[0])
            else:
                unique_graphs = {d.graph.unique_key(): d.graph for d in data}
                graph_plans = {k: planner.plan_random(g, 1)[0] for k, g in unique_graphs.items()}
                for d in data:
                    d.set_plan(graph_plans[d.graph.unique_key()])

            self.coverage()

        for d in data:
            plan, hyp, _ = fallback[d.graph.unique_key()]
            d.set_plan(plan).set_hyp(hyp)

        return self

    def post_process(self, reg: REG):
        entities = self.entities if hasattr(self, 'entities') else {}

        ents_reg_map = Counter()

        def process(text: str, seen=False):
            new_text, ents_map = reg.generate(text, entities)
            if seen:
                for t in ents_map:
                    ents_reg_map[t] += 1
            return new_text

        self.ents_reg_map = ents_reg_map
        self.data = [d.set_hyp(process(d.hyp, d.info["seen"])) for d in self.data]
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

    def single_coverage(self, plan, hyp):
        ents = False
        order = False

        if hyp is not None:
            p_ents = get_entities(plan)
            h_ents = get_entities(hyp)

            if len(set(p_ents)) == len(set(h_ents)):
                ents = True
                if all([p == h for p, h in zip(p_ents, h_ents)]):
                    order = True

            # if not ents:
            #     print("\n\n")
            #     print("Failed ents")
            #     print(plan)
            #     print(hyp)
            #     print("\n\n")

        return ents, order

    def coverage(self):
        pairs = {"seen": {}, "unseen": {}}
        for d in self.data:
            pairs["seen" if d.info["seen"] else "unseen"][d.plan] = d.hyp

        coverage = {}
        for t, v in pairs.items():
            entities = 0
            order = 0
            for plan, hyp in v.items():
                e, o = self.single_coverage(plan, hyp)
                if e:
                    entities += 1
                if o:
                    order += 1

            coverage[t] = {"entities": out_of(entities, len(v)), "order": out_of(order, entities)}

        print("coverage", coverage)

        return coverage

    def retries(self):
        pairs = {"seen": {}, "unseen": {}}
        for d in self.data:
            pairs["seen" if d.info["seen"] else "unseen"][d.plan] = d.plan_changes - 1 if hasattr(d,
                                                                                                  "plan_changes") else 1

        sums = {k: np.average(list(v.values())) for k, v in pairs.items()}
        print("sums", sums)
        return sums

    def for_manual_evaluation(self):
        graphs = {}
        for d in self.data:
            if "manual" in d.info and d.info["manual"]:
                graphs[d.graph.unique_key()] = {
                    "id": d.info["id"] if hasattr(d.info, "id") else None,
                    "sen": d.hyp,
                    "rdf": [(r, d, f, None) for r, d, f in d.graph.as_rdf()],
                    "hal": 0
                }

        return list(graphs.values())

    def describe_entities(self):
        return self

    def export(self):
        return [{"rdf": d.graph.as_rdf(), "text": d.text, "delex": d.delex, "plan": d.plan} for d in self.data]
