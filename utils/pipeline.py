import pickle
from os import path

from utils.file_system import makedir
from utils.silencer import Silencer
from utils.time import Time

cache_dir = path.join(path.dirname(path.abspath(__file__)), path.pardir, "cache")
print("Cache Directory", cache_dir, "\n")


class Pipeline:
    def __init__(self, params=None, mute=False):
        self.queue = []

        self.params = params if params else {}
        self.mute = mute
        self.local_timer = None
        self.global_timer = None

    def mutate(self, params=None):
        new = Pipeline(params if params else self.params)
        new.queue = list(self.queue)
        return new

    def enqueue(self, key, name, method):
        self.queue.append((key, name, method))

    def execute(self, run_name=None, tabs=0, x_params=None, previous_name=None):
        self.local_timer = Time.now()
        self.global_timer = Time.now()

        if not x_params:
            x_params = {}

        if not previous_name:
            previous_name = cache_dir

        makedir(previous_name)

        if run_name:
            print("\t" * tabs, run_name)

        key_len = max([len(k) for k, n, m in self.queue] + [0]) + 5
        name_len = max([len(n) for k, n, m in self.queue] + [0]) + 5

        for key, name, method in self.queue:
            if key != "out" and not isinstance(method, Pipeline):
                print(("\t" * (tabs + 1)) + ("%-" + str(key_len) + "s %-" + str(name_len) + "s") % (key, name), end=" ")

            pn = path.join(previous_name, key)
            pnf = pn + ".sav"

            if key != "out" and path.isfile(pnf):
                f = open(pnf, "rb")
                self.params[key] = pickle.load(f)
                f.close()
            else:
                if isinstance(method, Pipeline):
                    self.params[key] = method.execute(run_name=name, tabs=tabs + 1,
                                                      x_params={**x_params, **self.params}, previous_name=pn)
                    if "out" in self.params[key]:
                        self.params[key] = self.params[key]["out"]
                else:
                    if self.mute:
                        Silencer.mute()
                    self.params[key] = method(self.params, x_params)
                    if self.mute:
                        Silencer.unmute()

                    f = open(pnf, "wb")
                    pickle.dump(self.params[key], f)
                    f.close()

            if key != "out" and not isinstance(method, Pipeline):
                local_passed, global_passed = self.timer_report()
                report = self.params[key].report() if hasattr(self.params[key], "report") else ""
                print(("%-15s\t%-15s\t\t" + report) % (local_passed, global_passed))

        return self.params

    def timer_report(self):
        local_passed = Time.passed(self.local_timer)
        global_passed = Time.passed(self.global_timer)
        self.local_timer = Time.now()
        return local_passed, global_passed


ParallelPipeline = Pipeline
