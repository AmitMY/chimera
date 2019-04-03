import os
import re
import shutil
import subprocess
import sys
from itertools import chain
from os import path
from typing import List

import torch

from model.model_runner import ModelRunner, Model, add_features
from utils.file_system import save_temp, temp_name, listdir, temp_dir, save_temp_bin

is_cuda = torch.cuda.is_available()

libDirectory = path.join(path.dirname(path.realpath(__file__)), os.pardir, "libs", "OpenNMT")


def run_param(script, params):
    args = chain.from_iterable([["-" + k, v] for k, v in params.items()])
    args = list(map(str, filter(lambda a: a != None, args)))
    all_args = [sys.executable, path.join(libDirectory, script)] + args
    print("EXEC", " ".join(all_args))

    subprocess.run(all_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_entities(text: str):
    return list(re.findall("ent_(.*?)_ent", text, flags=re.IGNORECASE))


def find_best_out(plan, outs):
    plan_entities = set(get_entities(plan))
    possibilities = []
    for out in outs:
        out_entities = set(get_entities(out))
        if len(plan_entities) == len(out_entities):
            return out

        possibilities.append([out, len(out_entities)])

    return max(possibilities, key=lambda o: o[1])[0]


BEAM_SIZE = 10


class OpenNMTModel(Model):
    def __init__(self, model):
        self.model_bin = model

    def run_traslate(self, model_path, input_file, output_file, opt):
        opt["model"] = model_path
        opt["src"] = input_file
        opt["output"] = output_file
        if is_cuda:
            opt["gpu"] = 0

        run_param('translate.py', opt)

    def translate(self, plans: List[str], opts=None):  # Translate entire reader file using a model
        if not opts:
            opts = {
                "beam_size": BEAM_SIZE,
                "find_best": True
            }

        model_path = save_temp_bin(self.model_bin)

        o_lines = [[add_features(s.strip()) for i, s in enumerate(s.split("."))] if s != "" else [] for s in plans]
        n_lines = list(set(chain.from_iterable(o_lines)))

        if len(n_lines) == 0:
            return []

        source_path = save_temp(n_lines)
        target_path = temp_name()

        n_best = opts["beam_size"] if opts["find_best"] else 1

        self.run_traslate(model_path, source_path, target_path, {
            "replace_unk": None,
            "beam_size": opts["beam_size"],
            "n_best": n_best,
            "batch_size": 64
        })

        out_lines_f = open(target_path, "r", encoding="utf-8")
        out_lines = chunks(out_lines_f.read().splitlines(), n_best)
        out_lines_f.close()

        map_lines = {n: find_best_out(n, out) for n, out in zip(n_lines, out_lines)}

        return [" ".join([map_lines[s] for s in lines]) for lines in o_lines]


class OpenNMTModelRunner(ModelRunner):
    def __init__(self, train_reader, dev_reader):
        super().__init__(train_reader=train_reader, dev_reader=dev_reader)

    def pre_process(self):
        save_data = temp_dir()

        train_src = save_temp([add_features(d.plan) for d in self.train_reader.data])
        train_tgt = save_temp([d.delex for d in self.train_reader.data])
        valid_src = save_temp([add_features(d.plan) for d in self.dev_reader.data])
        valid_tgt = save_temp([d.delex for d in self.dev_reader.data])

        run_param('preprocess.py', {
            "train_src": train_src,
            "train_tgt": train_tgt,
            "valid_src": valid_src,
            "valid_tgt": valid_tgt,
            "save_data": save_data + "data",
            "dynamic_dict": None  # This will add a dynamic-dict parameter
        })

        data_zip = shutil.make_archive(base_name=temp_name(), format="gztar", root_dir=save_data)

        f = open(data_zip, "rb")
        bin_data = f.read()
        f.close()
        return bin_data

    def train(self, save_data, opt):
        save_data_archive = save_temp_bin(save_data)

        save_data_dir = temp_dir()
        shutil.unpack_archive(filename=save_data_archive, extract_dir=save_data_dir, format="gztar")

        save_model = temp_dir()

        opt["data"] = save_data_dir + "data"
        opt["save_model"] = save_model
        if is_cuda:
            opt["world_size"] = 1
            opt["gpu_ranks"] = 0

        run_param('train.py', opt)

        return save_model

    def find_best(self, checkpoints_dir):
        def checkpoint_number(checkpoint):
            return int(checkpoint.split(".")[0].split("_")[-1])

        checkpoints = list(sorted(listdir(checkpoints_dir), key=lambda f: checkpoint_number(f)))

        max_dev = {"score": 0, "model": None}
        dev_scores = []
        for model_path in checkpoints:
            f = open(model_path, "rb")
            model = OpenNMTModel(f.read())
            f.close()

            bleu = model.evaluate(self.dev_reader)[0]
            print("BLEU", bleu)

            dev_scores.append(bleu)
            if bleu > max_dev["score"]:
                max_dev = {"score": bleu, "model": model}

        print("Max Dev", max_dev)
        print("Scores", dev_scores)

        return max_dev["model"]
