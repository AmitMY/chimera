import pickle
import random
from os import path

from utils.file_system import makedir
from utils.silencer import Silencer
from utils.time import Time

cache_dir = path.join(path.dirname(path.abspath(__file__)), path.pardir, "cache")


print(cache_dir)

# TODO create CachedValue so in a function like "out" we can call CachedValue(f, "model")
class CachedDict:
    loaded_cache = {}

    def __init__(self, initial_dict={}):
        self.val_dict = dict(initial_dict)
        self.cache_dict = {}

    def add_cache(self, key: str, cache_location: str):
        self.cache_dict[key] = cache_location

    def load_cache(self, key: str):
        cache_location = self.cache_dict[key]
        if cache_location not in CachedDict.loaded_cache:
            f = open(cache_location, "rb")
            ext = cache_location.split(".")[-1]
            CachedDict.loaded_cache[cache_location] = pickle.load(f) if ext == "sav" else f.read()
            f.close()
        self.val_dict[key] = CachedDict.loaded_cache[cache_location]
        del self.cache_dict[key]
        return self.val_dict[key]

    def copy_key(self, self_key, other_cache, other_key):
        if other_key in other_cache.val_dict:
            self.val_dict[self_key] = other_cache.val_dict[other_key]
        else:
            self.cache_dict[self_key] = other_cache.cache_dict[other_key]

    def union(self, other_cached_dict):
        new = CachedDict()
        new.val_dict = {**self.val_dict, **other_cached_dict.val_dict}
        new.cache_dict = {**self.cache_dict, **other_cached_dict.cache_dict}
        return new

    def keys(self):
        return list(self.val_dict.keys()) + list(self.cache_dict.keys())

    def __getitem__(self, key: str):
        if key in self.cache_dict:
            return self.load_cache(key)
        if key in self.val_dict:
            return self.val_dict[key]
        raise KeyError("Key " + str(key) + " Not found in values nor cache")

    def __setitem__(self, key, value):
        self.val_dict[key] = value

    def __contains__(self, key):
        return key in self.val_dict or key in self.cache_dict


class QueueItem:
    def __init__(self, key, name, method, ext="sav", load_cache=True, load_self=False):
        self.key = key
        self.name = name
        self.method = method
        self.ext = ext
        self.load_cache = load_cache
        self.load_self = load_self


class Pipeline:
    def __init__(self, params=None, mute=False, key=None):
        self.queue = []

        self.params = CachedDict().union(params) \
            if isinstance(params, CachedDict) else CachedDict(params if params else {})
        self.mute = mute
        self.local_timer = None
        self.global_timer = None
        self.key = key

    def mutate(self, params=None):
        new = Pipeline(params if params else self.params)
        new.queue = list(self.queue)
        return new

    def enqueue(self, key, name, method, ext="sav", load_cache=True, load_self=False):
        self.queue.append(QueueItem(key, name, method, ext, load_cache, load_self))

    def execute(self, run_name=None, tabs=0, x_params=None, previous_name=None, cache_name: str = None):
        self.local_timer = Time.now()
        self.global_timer = Time.now()

        if not x_params:
            x_params = CachedDict()

        if not previous_name:
            previous_name = cache_dir

        makedir(previous_name)

        if cache_name:
            previous_name = path.join(previous_name, cache_name)
            makedir(previous_name)

        if run_name:
            print("  " * tabs, run_name)

        key_len = max([len(qi.key) for qi in self.queue] + [0]) + 5
        name_len = max([len(qi.name) for qi in self.queue] + [0]) + 5

        for qi in self.queue:
            # key, name, method, load_cache, load_self
            if qi.key != "out" and not isinstance(qi.method, Pipeline):
                print(("  " * (tabs + 1)) + ("%-" + str(key_len) + "s %-" + str(name_len) + "s") % (qi.key, qi.name),
                      end=" ")

            pn = path.join(previous_name, qi.key)
            pnf = pn + "." + qi.ext

            if qi.load_cache and qi.key != "out" and path.isfile(pnf):
                self.params.add_cache(qi.key, pnf)
                if qi.load_self:
                    self.params.load_cache(qi.key)
            else:
                if isinstance(qi.method, Pipeline):
                    self.params[qi.key] = qi.method.execute(run_name=qi.name, tabs=tabs + 1,
                                                            x_params=x_params.union(self.params), previous_name=pn)
                    if "out" in self.params[qi.key]:
                        self.params.copy_key(qi.key, self.params[qi.key], "out")
                else:
                    if self.mute:
                        Silencer.mute()
                    self.params[qi.key] = qi.method(self.params, x_params)
                    if self.mute:
                        Silencer.unmute()

                    f = open(pnf, "wb")
                    if qi.ext == "sav":
                        pickle.dump(self.params[qi.key], f)
                    else:
                        f.write(self.params[qi.key])
                    f.close()

            if qi.key != "out" and not isinstance(qi.method, Pipeline):
                local_passed, global_passed = self.timer_report()
                report = self.params[qi.key].report() \
                    if qi.key in self.params.val_dict and hasattr(self.params[qi.key], "report") else ""
                print(("%-15s\t\t" + report) % (local_passed))

        return self.params

    def timer_report(self):
        local_passed = Time.passed(self.local_timer)
        global_passed = Time.passed(self.global_timer)
        self.local_timer = Time.now()
        return local_passed, global_passed


class ShuffledPipeline(Pipeline):
    def execute(self, **kwargs):
        random.shuffle(self.queue)
        return super().execute(**kwargs)


ParallelPipeline = Pipeline
