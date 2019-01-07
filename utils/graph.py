import json
import re
from collections import defaultdict
from enum import Enum
from itertools import chain, product, permutations, combinations

from utils.delex import concat_entity
from utils.time import Time


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


class StructuredNode:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children
        self.lins = None

    def get_val(self, concat_entities=False):
        return concat_entity(self.value)

    def linearizations(self):
        if self.value == NodeType.FILTER_OUT:
            return []
        if not self.lins:
            self.lins = self.rec_linearizations()
        return self.lins

    def rec_linearizations(self):
        if len(self.children) == 0:
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


class Graph:
    def __init__(self, rdfs=[]):
        self.graph = defaultdict(list)
        self.edges = {}
        self.undirected_edges = {}
        self.nodes = set()

        for s, r, o in rdfs:
            self.add_edge(s, o, r)

    def add_edge(self, s, o, l):
        self.graph[s].append(o)
        self.edges[(s, o)] = l
        self.undirected_edges[(s, o)] = "> " + l
        self.graph[o].append(s)
        self.undirected_edges[(o, s)] = "< " + l

        self.nodes.add(s)
        self.nodes.add(o)

    def as_rdf(self):
        return [(n1, e, n2) for ((n1, n2), e) in self.edges.items()]

    def unique_key(self):
        return tuple(self.as_rdf())

    def exhaustive_plan(self):
        return self.sub_graphs_plan()

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

    def sub_graphs_plan(self, max_size=4, plan_cache=None, graph_plan_cache=None):
        if not plan_cache:
            plan_cache = {}
        if not graph_plan_cache:
            graph_plan_cache = {}

        rdfs = self.as_rdf()
        sub_graphs = [list(p) for p in powerset(rdfs) if max_size >= len(p) > 0 and len(p) != len(rdfs)]

        g_s = json.dumps(rdfs)
        if g_s not in plan_cache:
            plan_cache[g_s] = self.plan_all()
        options = [plan_cache[g_s]]

        for g1 in sub_graphs:
            g2 = [r for r in rdfs if r not in g1]

            g1_s = json.dumps(g1)
            g2_s = json.dumps(g2)

            if g1_s not in plan_cache:
                plan_cache[g1_s] = Graph(g1).plan_all()
            if g2_s not in graph_plan_cache:
                graph_plan_cache[g2_s] = Graph(g2).sub_graphs_plan(max_size, plan_cache, graph_plan_cache)

            options.append(
                StructuredNode(NodeType.SENTENCES, [("", plan_cache[g1_s]), ("", graph_plan_cache[g2_s])]))

        return StructuredNode(NodeType.OR, [("", o) for o in options])

    def plan_all(self):
        return StructuredNode(NodeType.OR, [("", self.plan_from(node)) for node in self.nodes])

    def plan_from(self, node):
        visited = set()
        plan = self.dfs(node, visited)

        # If there is more than 1 connected component
        if len(visited) != len(self.nodes):
            return StructuredNode(NodeType.FILTER_OUT)

        return plan

    def dfs(self, node, visited=None):
        if not isinstance(visited, set):
            visited = set()

        if node in visited:
            return None

        visited.add(node)

        children = [(self.undirected_edges[(node, n)], self.dfs(n, visited)) for n in self.graph[node]]
        children = [(readable_edge(e), n) for (e, n) in children if n]  # Filter out Nones.

        children = [("", StructuredNode(NodeType.AND, children))] if len(children) > 0 else []

        return StructuredNode(node, children)


if __name__ == "__main__":
    g = Graph()
    g.add_edge('A', 'B', 'b')
    g.add_edge('A', 'C', 'c')
    g.add_edge('B', 'D', 'd')
    g.add_edge('D', 'E', 'e')

    plans = g.exhaustive_plan().linearizations()
    print("exhaustive", len(plans))

    plans = g.constraint_graphs_plan([
        {'must_include': set({"A", "B"}), 'must_exclude': set({})},
        {'must_include': set({"A", "C"}), 'must_exclude': set({})},
        {'must_include': set({"B", "D", "E"}), 'must_exclude': set({})}]).linearizations()

    print("constraint", len(plans))

    for p in plans:
        print(p)

    print()

    rdfs = [('William_Anders', 'dateOfRetirement', '"1969-09-01"'), ('William_Anders', 'was selected by NASA', '1963'),
            ('William_Anders', 'timeInSpace', '"8820.0"(minutes)'), ('William_Anders', 'birthDate', '"1933-10-17"'),
            ('William_Anders', 'occupation', 'Fighter_pilot'), ('William_Anders', 'birthPlace', 'British_Hong_Kong'),
            ('William_Anders', 'was a crew member of', 'Apollo_8')]
    s = Graph(rdfs)
    # s.add_edge('A', 'B', 'b')
    # s.add_edge('A', 'C', 'c')
    # s.add_edge('A', 'D', 'd')
    # s.add_edge('A', 'E', 'e')
    # s.add_edge('A', 'F', 'f')
    # s.add_edge('A', 'G', 'g')
    # s.add_edge('A', 'H', 'h')
    # s.add_edge('A', 'I', 'i')
    # s.add_edge('A', 'J', 'j')
    # s.add_edge('A', 'K', 'k')

    print("exhaustive")
    now = Time.now()
    plans = s.exhaustive_plan().linearizations()
    print(len(plans), "plans")
    print(Time.passed(now))

    print("constraint")
    now = Time.now()
    plans = s.constraint_graphs_plan([
        {'must_include': set({"B", "C", "D"}), 'must_exclude': set({})},
        {'must_include': set({"E"}), 'must_exclude': set({})},
        {'must_include': set({"F", "G"}), 'must_exclude': set({})}]).linearizations()
    print(len(plans), "plans")
    print(Time.passed(now))
