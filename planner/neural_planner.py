import re
from typing import Tuple, List

import dynet_config

dynet_config.set(autobatch=1, mem="2048")


from itertools import chain

import dynet as dy
import numpy as np

from data.reader import DataReader
from eval.bleu.eval import BLEU
from planner.planner import Planner
from scorer.scorer import get_relations
from utils.delex import concat_entity
from utils.dynet_model_executer import Vocab, DynetModelExecutor, BaseDynetModel, arg_sample
from utils.graph import Graph, readable_edge
from utils.tokens import tokenize


class Model(BaseDynetModel):
    def __init__(self, embedding_size=20, entity_dropout=0.3, relation_dropout=0.3, max_edges=10):
        super().__init__()

        self.vocab = None
        self.embedding_size = embedding_size
        self.counter_size = 5

        self.entity_dropout = entity_dropout
        self.relation_dropout = relation_dropout

        self.decoder = None

        self.counters = Vocab(list(range(max_edges + 1)))

    def set_vocab(self, in_vocab: Vocab, out_vocab: Vocab):
        self.vocab = out_vocab  # Use same vocab for both the input and the output

    def init_params(self):
        super().init_params()

        self.entity_encoder = self.pc.add_parameters((self.embedding_size, self.embedding_size * 3))  # e N e
        self.relation_encoder = self.pc.add_parameters((self.embedding_size, self.embedding_size * 3))  # N e N
        self.no_ent = self.pc.add_parameters(self.embedding_size)

        self.vocab.create_lookup(self.pc, self.embedding_size)
        self.counters.create_lookup(self.pc, self.counter_size)
        self.decoder = dy.LSTMBuilder(3, self.embedding_size + self.counter_size * 4, self.embedding_size, self.pc)

    def fix_out(self, plan: str):
        if not plan:
            return None

        for d, r in get_relations(plan):
            plan = plan.replace(d + ' ' + r + ' [ ', d + '_' + r + '_')

        return plan.split(" ")

    def eval(self, predictions, truth):
        print("predictions", predictions[-1])
        predictions = [" ".join(chain.from_iterable(p)) for p in predictions]
        print("predictions", predictions[-1])
        print("truth", truth[-1])
        return BLEU(predictions, truth, single_ref=True)[0]

    def forward(self, g: Graph, out: str = None, greedy=True):
        out_tokens = self.fix_out(out)

        # Encoding
        nodes = {n: self.vocab.lookup(self.word_dropout(n, self.entity_dropout if out else 0))
                 for n in g.nodes}
        unique_edges = set(chain.from_iterable(g.edges.values()))
        edges = {e: self.vocab.lookup(self.word_dropout(e, self.relation_dropout if out else 0))
                 for e in unique_edges}

        node_connections = {node: ([], []) for node in g.nodes}
        for ((n1, n2), es) in g.edges.items():
            for e in es:
                edge_rep = edges[e]
                node_connections[n2][0].append(edge_rep)
                node_connections[n1][1].append(edge_rep)

        ne_rep = lambda e: dy.average(e) if len(e) > 0 else self.no_ent
        node_reps = {n: self.entity_encoder *
                        dy.concatenate([ne_rep(node_connections[n][0]), nodes[n], ne_rep(node_connections[n][1])])
                     for n in g.nodes}

        edge_reps = [self.relation_encoder * dy.concatenate([node_reps[n1], edges[e], node_reps[n2]])
                     for ((n1, n2), es) in g.edges.items() for e in es]

        # In decoding time we will remove 1 RDF at a time until none is left.
        rdfs = {((n1, n2), e): edge_reps[i] for i, ((n1, n2), es) in enumerate(g.edges.items()) for e in es}

        # Decoding
        nodes_stack = []
        edges_coverage = {"yes": 0, "no": len(g.edges), "current": 0}
        counter_vec = lambda: dy.concatenate([self.counters.lookup(c) for c in
                                              [len(nodes_stack)] + list(edges_coverage.values())])

        c_vec = counter_vec()
        initial_input = dy.concatenate([dy.average(edge_reps), c_vec])
        decoder = self.decoder.initial_state().add_input(initial_input)

        def choose(item):
            if out_tokens:
                out_tokens.pop(0)

            if item[0] == "pop":
                nodes_stack.pop()
                res = [item[1]]
                if len(nodes_stack) == 0:
                    edges_coverage["current"] = 0
            elif item[0] == "node":
                nodes_stack.append(item[1])
                res = [item[1]]
            elif item[0] == "edge":
                edges_coverage["yes"] += 1
                edges_coverage["current"] += 1
                edges_coverage["no"] -= 1

                _, d, e, n = item
                prev_node = nodes_stack[-1]
                nodes_stack.append(n)
                res = [d, e, "[", n]
                if d == ">":
                    del rdfs[(prev_node, n), e]
                elif d == "<":
                    del rdfs[(n, prev_node), e]
                else:
                    raise ValueError("direction can only be > or <. got " + d)
            else:
                raise ValueError("type can only be: pop, node, edge. got " + item[0])

            c_vec = counter_vec()
            for w in res:
                if w in node_reps:
                    vec = node_reps[w]
                elif w in edges:
                    vec = edges[w]
                else:
                    vec = self.vocab.lookup(w)
                decoder.add_input(dy.concatenate([vec, c_vec]))

            return res

        is_pop = False
        while len(rdfs) > 0:
            # Possible vocab
            if len(nodes_stack) == 0:
                is_pop = False
                vocab = {("node", n): node_reps[n] for n in
                         set(chain.from_iterable([ns for ns, e in rdfs.keys()]))}
            else:
                last_node = nodes_stack[-1]
                f_edges = {("edge", ">", e, n2): rep for ((n1, n2), e), rep in rdfs.items() if n1 == last_node}
                b_edges = {("edge", "<", e, n1): rep for ((n1, n2), e), rep in rdfs.items() if n2 == last_node}
                vocab = {**f_edges, **b_edges}

                if is_pop:
                    # What node are we popping to. To help neighboring facts
                    pop_node = self.no_ent if len(nodes_stack) == 1 else node_reps[nodes_stack[-2]]
                    pop_char = "." if len(nodes_stack) == 1 else "]"
                    vocab[("pop", pop_char)] = dy.esum([self.vocab.lookup(pop_char), pop_node])
                is_pop = True  # next iteration is popable

            vocab_list = list(vocab.items())
            vocab_index = ["_".join(i[1:]) for i, _ in vocab_list]

            try:
                if len(vocab_list) == 1:
                    if out:
                        assert out_tokens[0] == vocab_index[0]

                    choice = choose(vocab_list[0][0])
                    if not out:
                        yield choice

                    continue

                vocab_matrix = dy.transpose(dy.concatenate_cols([rep for _, rep in vocab_list]))
                pred_vec = vocab_matrix * decoder.output()
                if out:
                    best_i = vocab_index.index(out_tokens[0])
                    choose(vocab_list[best_i][0])
                    yield dy.pickneglogsoftmax(pred_vec, best_i)
                else:
                    if greedy:
                        best_i = int(np.argmax(pred_vec.npvalue()))
                    else:
                        best_i = arg_sample(list(dy.softmax(pred_vec).npvalue()))

                    yield choose(vocab_list[best_i][0])
            except Exception as e:
                print()
                print("is_pop", is_pop)
                print("out", out)
                print("out tokens", out_tokens)
                print("vocab_index", vocab_index)
                print("original_rdf", g.as_rdf())
                print("rdf", list(rdfs.keys()))
                print()
                raise e

        if not out:
            yield ["]"]


class NeuralPlanner(Planner):
    re_plan = True

    def __init(self):
        self.executor = None

    def convert_relation(self, r: str):
        return "_".join(tokenize(readable_edge(r)))

    def convert_graph(self, g: Graph):
        rdf = [(concat_entity(s), self.convert_relation(r), concat_entity(o)) for s, r, o in g.as_rdf()]
        return Graph(rdf)

    def convert_plan(self, p: str):
        relations = get_relations(p)
        for d, r in relations:
            p = p.replace(r, self.convert_relation(r))

        while "]]" in p:
            p = p.replace("]]", "] ]")

        p = p.replace("].", "] .")

        return re.sub("\[(\w)", r"[ \1", p)

    def convert_set(self, reader: DataReader):
        return [(self.convert_graph(d.graph), self.convert_plan(d.plan)) for d in reader.copy().data]

    def learn(self, train_reader: DataReader, dev_reader: DataReader):
        train_set = self.convert_set(train_reader)
        dev_set = self.convert_set(dev_reader)

        model = Model()
        self.executor = DynetModelExecutor(model, train_set, dev_set)
        for batch_exponent in range(0, 3):
            self.executor.train(5, batch_exponent)

        return self

    def score(self, g: Graph, plan: str):
        error = self.executor.calc_error(self.convert_graph(g), self.convert_plan(plan))
        return 1 / error  # To make less error score better

    def model_plan(self, g: Graph, greedy=True):
        predict = list(self.executor.predict([self.convert_graph(g)], greedy=greedy))[0]
        plan = " ".join(chain.from_iterable(predict))
        for d, r in get_relations(plan):
            plan = plan.replace(d + " " + r, d + " " + r.replace("_", " "))
        return plan

    def plan_random(self, g: Graph, amount: int):
        return [self.model_plan(g, greedy=False) for _ in range(amount)]

    def plan_best(self, g: Graph, ranker_plans=None):
        if ranker_plans:
            raise NotImplementedError("Planner.plan_best is not implemented when ranker_plans is defined")

        return self.model_plan(g, greedy=True)
