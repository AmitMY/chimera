from itertools import chain

import numpy as np
import random
import time
from typing import Callable, Tuple, List

import dynet as dy
from tqdm import tqdm

from utils.file_system import temp_name, save_temp_bin

# Vocabulary for model
from utils.graph import Graph


class Vocab:
    def __init__(self, words, vectors=None, is_unique=False, update=True):
        # This option is to preserve order in case order is needed
        if not is_unique:
            words = list(set(words))

        # Update pre-trained vectors
        self.update = update
        # Pre-trained vectors
        self.vectors = vectors

        # Word to index dictionary (encode)
        self.w2i = {w: i for i, w in enumerate(words)}
        # Index to word dictionary (decode)
        self.i2w = {i: w for w, i in self.w2i.items()}
        # Unknown vectors
        self.UNK = None
        self.UNKNumber = None
        # Model lookup table
        self.lookup_table = None

    def create_lookup(self, pc, size, init=None):
        self.UNK = pc.add_parameters(size)
        self.UNKNumber = pc.add_parameters(size)
        if not init:
            self.lookup_table = pc.add_lookup_parameters((len(self.w2i), size))
        else:
            self.lookup_table = pc.add_lookup_parameters((len(self.w2i), size), init=init)
        if self.vectors:
            init = [None for _ in range(len(self.w2i))]
            for w, i in self.w2i.items():
                init[i] = self.vectors[w]
            self.lookup_table.init_from_array(np.array(init))

    def lookup(self, word):
        if word in self.w2i:
            return dy.lookup(self.lookup_table, self.w2i[word], update=self.update)

        try:
            float(word)
            return self.UNKNumber
        except ValueError:
            return self.UNK

    def __getstate__(self):
        d = {**self.__dict__}
        for k, v in self.__dict__.items():
            if "_dynet." in str(type(v)):
                del d[k]
        return d

    def __len__(self):
        return len(self.w2i)


def arg_sample(probs: List[float]):
    rand = random.random()
    # print(probs)
    for i, p in enumerate(probs):
        if rand <= p:
            return i
        rand -= p

    raise ValueError("probs doesn't sum to 1?")


class BaseDynetModel:
    def init_params(self):
        self.pc = dy.ParameterCollection()

    def word_dropout(self, word: str, dropout: float):
        return word if random.uniform(0, 1) > dropout else "Unknown Token"

    def __getstate__(self):
        d = {**self.__dict__}
        if "pc" in d:
            temp = temp_name()
            self.pc.save(temp)
            d["recovery"] = open(temp, "rb").read()

        for k, v in self.__dict__.items():
            if "_dynet." in str(type(v)):
                del d[k]

        return d

    def __setstate__(self, state):
        for k, v in state.items():
            self.__dict__[k] = v

        if "recovery" in state:
            recovery = save_temp_bin(state["recovery"])
            del self.__dict__["recovery"]
            self.init_params()
            self.pc.populate(recovery)


class DynetModelExecutor:
    def __init__(self, model, train_data: List[Tuple], dev_data: List[Tuple]):
        self.model = model

        self.train_data = train_data
        self.dev_data = dev_data

        self.results = []  # Dev results over time
        self.losses = []  # Loss over time

        self.snapshot = None

        # build vocabs
        _in = Vocab(chain.from_iterable([str(g).split(" ") for g, p in train_data]))
        _out = Vocab(chain.from_iterable([str(p).split(" ") for g, p in train_data]))
        self.model.set_vocab(_in, _out)

        self.model.init_params()
        self.init_params()

    def init_params(self):
        self.trainer = dy.AdamTrainer(self.model.pc)

    @staticmethod
    def batch(iterable, n=1):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]

    def calc_error(self, _in, _out):
        return self.calc_errors([(_in, _out)])[0]

    def calc_errors(self, batch: List[Tuple]):
        dy.renew_cg()
        errors_exp = dy.concatenate([dy.average(list(self.model.forward(_in, _out))) for _in, _out in batch])
        errors = errors_exp.value()
        if len(batch) == 1:
            errors = [errors]
        return np.array(errors)

    def train_epoch(self, batch_size):
        batches = list(self.batch(self.train_data, batch_size))
        for batch in tqdm(batches, unit="batch-" + str(batch_size)):
            dy.renew_cg()
            error = dy.esum([dy.average(list(self.model.forward(_in, _out))) for _in, _out in batch])

            self.losses.append(float(error.value()) / len(batch))

            error.backward()
            self.trainer.update()

        time.sleep(0.01)

        return sum(self.losses[-1 * len(batches):]) / len(batches)

    def train(self, epochs, batch_exponent=2):
        for i in range(1, epochs + 1):
            print("Epoch:", i)
            random.shuffle(self.train_data)

            batch_size = 2 ** batch_exponent

            loss = self.train_epoch(batch_size)

            print("Loss", loss)

            previously_best_dev = max(self.results) if len(self.results) > 0 else 0

            dev_in, dev_out = zip(*self.dev_data)

            self.results.append(self.model.eval(list(self.predict(dev_in)), dev_out))
            print("Dev", self.results[-1])

            # if self.results[-1] >= previously_best_dev:
            #     self.save()

            print("Best Dev", max(self.results))

            time.sleep(0.01)

    def predict(self, inputs, greedy=True):
        for _in in inputs:
            dy.renew_cg()
            results = list(self.model.forward(_in, greedy=greedy))

            yield results

    def __getstate__(self):
        d = {**self.__dict__}
        for k, v in self.__dict__.items():
            if "_dynet." in str(type(v)):
                del d[k]

        return d

    def __setstate__(self, state):
        for k, v in state.items():
            self.__dict__[k] = v
        self.init_params()
