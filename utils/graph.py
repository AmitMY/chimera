import json
import pickle
import re
import sys
import zlib
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from itertools import chain, product, permutations, combinations
from typing import Set, List

from tqdm import tqdm

from utils.delex import concat_entity
from utils.memoize import memoize
from utils.time import Time
import psutil


@lru_cache(maxsize=None)
def readable_edge(e):
    return " ".join(re.sub('(?!^)([A-Z][a-z]+)', r' \1', e.replace("_", " ")).split()).lower()


def powerset(iterable):
    xs = list(iterable)
    return list(chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1)))


class NodeType(Enum):
    SENTENCES = "__SENTENCE__"
    AND = "__AND__"
    OR = "__OR__"
    FILTER_OUT = "__FILTER_OUT__"
    FINAL = "__FINAL__"


class StructuredNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children
        self.lins = None

    def get_val(self):
        return concat_entity(self.value)

    def linearizations(self):
        if self.value == NodeType.FILTER_OUT:
            return []
        if not self.lins:
            self.lins = self.rec_linearizations()
        return self.lins

    def rec_linearizations(self):
        if self.children is None:
            return [self.get_val()]

        edges = [[(e + " [" + l + "]") if e else l for l in s.linearizations()] for e, s in
                 self.children]

        if self.value == NodeType.OR:
            return [l for e in edges for l in e]

        edges = list(product(*edges))

        if self.value == NodeType.SENTENCES:
            return [". ".join(p) for p in edges]

        if self.value == NodeType.AND:
            return [" ".join(p) for e in edges for p in permutations(e)]

        return [self.get_val() + " " + " ".join(e) for e in edges]


class LinearNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

    def linearizations(self):
        none_empty = [s for s in self.rec_linearizations() if len(s) > 0]
        return [" ".join(s[:-1]) for s in none_empty if s[-1] != NodeType.FILTER_OUT]

    def rec_linearizations(self):
        if self.next is None:
            return [[self.value]]

        if self.value == NodeType.OR:
            return [l for n in self.next for l in n.rec_linearizations()]

        return [[self.value] + l for n in self.next for l in n.rec_linearizations()]


class Graph:
    def __init__(self, rdfs=[]):
        self.graph = defaultdict(list)
        self.edges = defaultdict(list)
        self.undirected_edges = defaultdict(list)
        self.nodes = set()

        for s, r, o in rdfs:
            self.add_edge(s, o, r)

    def add_edge(self, s, o, l):
        self.graph[s].append(o)
        self.edges[(s, o)].append(l)
        self.undirected_edges[(s, o)].append("> " + l)
        self.graph[o].append(s)
        self.undirected_edges[(o, s)].append("< " + l)

        self.nodes.add(s)
        self.nodes.add(o)

    def as_rdf(self):
        return [(n1, e, n2) for ((n1, n2), es) in self.edges.items() for e in es]

    def unique_key(self):
        return tuple(self.as_rdf())

    def exhaustive_plan(self, force_tree=False):
        return self.sub_graphs_plan(force_tree=force_tree)

    def constraint_graphs_plan(self, constraints):
        options = self.constraint_graphs_maker(constraints)
        if len(options) == 0:
            return StructuredNode(NodeType.FILTER_OUT, [])

        c = [("", StructuredNode(NodeType.SENTENCES, [("", g.plan_all()) for g in graphs])) for graphs in options]
        return StructuredNode(NodeType.OR, c)

    def constraint_graphs_maker(self, components, prev=None):
        if not prev:
            prev = []

        lengths = [len(self.nodes), len(components)]
        if all(map(lambda a: a == 0, lengths)):  # If both lengths are 0
            return [prev]

        if min(lengths) == 0 and max(lengths) > 0:  # If it cannot be satisfied
            return []

        comp = components[0]

        rdfs = self.as_rdf()

        options = []

        for g in map(list, powerset(rdfs)):
            if len(g) == 0:  # Skip empty graphs
                continue

            g_nodes = set(chain.from_iterable([[s, o] for s, r, o in g]))
            if comp["must_include"] <= g_nodes and len(comp["must_exclude"].intersection(g_nodes)) == 0:
                complement = Graph([r for r in rdfs if r not in g])
                options += complement.constraint_graphs_maker(components[1:], prev + [Graph(g)])

        return options

    def sub_graphs_plan(self, max_size=4, plan_cache=None, graph_plan_cache=None, force_tree=False):
        if not plan_cache:
            plan_cache = {}
        if not graph_plan_cache:
            graph_plan_cache = {}

        rdfs = self.as_rdf()
        sub_graphs = [list(p) for p in powerset(rdfs) if max_size >= len(p) > 0 and len(p) != len(rdfs)]

        g_s = json.dumps(rdfs)
        if g_s not in plan_cache:
            plan_cache[g_s] = self.plan_all(force_tree=force_tree)
        options = [plan_cache[g_s]]

        for g1 in sub_graphs:
            g2 = [r for r in rdfs if r not in g1]

            g1_s = json.dumps(g1)
            g2_s = json.dumps(g2)

            if g1_s not in plan_cache:
                plan_cache[g1_s] = Graph(g1).plan_all(force_tree=force_tree)
            if g2_s not in graph_plan_cache:
                graph_plan_cache[g2_s] = Graph(g2).sub_graphs_plan(max_size, plan_cache, graph_plan_cache)

            options.append(
                StructuredNode(NodeType.SENTENCES, [("", plan_cache[g1_s]), ("", graph_plan_cache[g2_s])]))

        return StructuredNode(NodeType.OR, [("", o) for o in options])

    def plan_all(self, force_tree=False):
        # If not a tree, very simple heuristic
        if not force_tree and any([len(es) > 1 for es in self.edges.values()]):
            return self.traverse_all()
        # More simple traversal only if tree
        return StructuredNode(NodeType.OR, [("", self.plan_from(node)) for node in self.nodes])

    def plan_from(self, node):
        visited = set()
        plan = self.dfs(node, visited)

        # If there is more than 1 connected component
        if len(visited) != len(self.nodes):
            return StructuredNode(NodeType.FILTER_OUT)

        return plan

    def dfs(self, node: str, visited: Set[str]):
        if node in visited:
            return None

        visited.add(node)

        children = [(self.undirected_edges[(node, n)], self.dfs(n, visited)) for n in self.graph[node]]
        children = [(readable_edge(e), n) for (es, n) in children for e in es if n]

        children = [("", StructuredNode(NodeType.AND, children))] if len(children) > 0 else []

        return StructuredNode(node, children)

    def traverse_all(self, nodes_stack=None, rdfs=None):
        if nodes_stack is None:
            rdfs = self.as_rdf()
            return LinearNode(NodeType.OR,
                              [LinearNode(concat_entity(n), self.traverse_all([n], rdfs)) for n in self.nodes])

        f_edges = [(i, n2, ">", e) for i, (n1, e, n2) in enumerate(rdfs) if n1 == nodes_stack[-1]]
        b_edges = [(i, n1, "<", e) for i, (n1, e, n2) in enumerate(rdfs) if n2 == nodes_stack[-1]]
        edges = f_edges + b_edges

        options = []
        for i, n, d, e in edges:
            new_rdfs = list(rdfs)
            new_rdfs.pop(i)
            text = " ".join([d, readable_edge(e), "[", concat_entity(n)])
            options.append(LinearNode(text, self.traverse_all(nodes_stack + [n], new_rdfs)))

        if len(nodes_stack) > 1:
            options.append(LinearNode("]", self.traverse_all(nodes_stack[:-1], rdfs)))

        if len(options) > 0:
            return options

        if len(rdfs) == 0:
            return [LinearNode(NodeType.FINAL)]
        return [LinearNode(NodeType.FILTER_OUT)]


class Compressor:
    def __init__(self):
        self.voc = {}

    def compress(self, plan: str):
        tokens = []
        for t in plan.split():
            if t not in self.voc:
                self.voc[t] = len(self.voc)
            tokens.append(chr(self.voc[t]))
        return "".join(tokens)


if __name__ == "__main__":
    g = Graph()
    g.add_edge('A', 'B', 'b1')
    # g.add_edge('A', 'B', 'b2')
    g.add_edge('A', 'C', 'c')
    g.add_edge('A', 'C', 'c2')
    # g.add_edge('C', 'C', 'cd')
    g.add_edge('A', 'D', 'd')
    g.add_edge('D', 'E', 'e')
    g.add_edge('A', 'E', 'e')

    now = Time.now()
    plans = list(g.exhaustive_plan(force_tree=False).linearizations())

    print("exhaustive_plan", len(plans))
    print(plans[0])
    print(plans[-1])
    print(Time.passed(now))

    print("memory size", sys.getsizeof(pickle.dumps(plans)) / 1024)
    plans_str = "\n".join(plans).encode("utf-8")
    compressed = zlib.compress(plans_str, 1)
    print("zlib size", sys.getsizeof(pickle.dumps(compressed)) / 1024)

    compressor = Compressor()
    plans = [compressor.compress(p) for p in plans]
    print("compressor size", sys.getsizeof(pickle.dumps({"p": plans, "c": compressor})) / 1024)

    plans_str = "\n".join(plans).encode("utf-8")
    print("compressor zlib size", sys.getsizeof(pickle.dumps({"p": zlib.compress(plans_str, 1), "c": compressor})) / 1024)


    # now = Time.now()
    # for i in range(1000):
    #     plans = g.plan_all().linearizations()
    # print("plan_all", len(plans))
    # for p in plans:
    #     print(p)
    # print(Time.passed(now))
    #
    # now = Time.now()
    # for i in range(1000):
    #     plans = g.traverse_all().linearizations()
    # print("traversal", len(plans))
    # for p in plans:
    #     print(p)
    # print(Time.passed(now))

    #
    # plans = g.constraint_graphs_plan([
    #     {'must_include': set({"A", "B"}), 'must_exclude': set({})},
    #     {'must_include': set({"A", "C"}), 'must_exclude': set({})},
    #     {'must_include': set({"B", "D", "E"}), 'must_exclude': set({})}]).linearizations()
    #
    # print("constraint", len(plans))
    #
    # for p in plans:
    #     print(p)
    #
    # print()
    #
    # rdfs = [('William_Anders', 'dateOfRetirement', '"1969-09-01"'), ('William_Anders', 'was selected by NASA', '1963'),
    #         ('William_Anders', 'timeInSpace', '"8820.0"(minutes)'), ('William_Anders', 'birthDate', '"1933-10-17"'),
    #         ('William_Anders', 'occupation', 'Fighter_pilot'), ('William_Anders', 'birthPlace', 'British_Hong_Kong'),
    #         ('William_Anders', 'was a crew member of', 'Apollo_8')]
    # s = Graph(rdfs)
    # # s.add_edge('A', 'B', 'b')
    # # s.add_edge('A', 'C', 'c')
    # # s.add_edge('A', 'D', 'd')
    # # s.add_edge('A', 'E', 'e')
    # # s.add_edge('A', 'F', 'f')
    # # s.add_edge('A', 'G', 'g')
    # # s.add_edge('A', 'H', 'h')
    # # s.add_edge('A', 'I', 'i')
    # # s.add_edge('A', 'J', 'j')
    # # s.add_edge('A', 'K', 'k')
    #
    # print("exhaustive")
    # now = Time.now()
    # plans = s.exhaustive_plan().linearizations()
    # print(len(plans), "plans")
    # print(Time.passed(now))
    #
    # print("constraint")
    # now = Time.now()
    # plans = s.constraint_graphs_plan([
    #     {'must_include': set({"B", "C", "D"}), 'must_exclude': set({})},
    #     {'must_include': set({"E"}), 'must_exclude': set({})},
    #     {'must_include': set({"F", "G"}), 'must_exclude': set({})}]).linearizations()
    # print(len(plans), "plans")
    # print(Time.passed(now))
