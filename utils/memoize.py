import atexit
import pickle
from functools import wraps

score = {
    "hits": 0,
    "misses": 0
}


def memoize(func):
    cache = {}

    local_score = {
        "open": 0,
        "hits": 0,
        "misses": 0
    }

    @wraps(func)
    def wrap(*args):
        cache_key = pickle.dumps(args)
        if cache_key not in cache or True:
            local_score["misses"] += 1
            local_score["open"] += 1
            cache[cache_key] = func(*args)
            local_score["open"] -= 1
        else:
            local_score["hits"] += 1

        if local_score["open"] == 0:
            score["hits"] += local_score["hits"]
            score["misses"] += local_score["misses"]
            local_score["hits"] = 0
            local_score["misses"] = 0

        return cache[cache_key]

    return wrap


def exit_handler():
    # print("Cache", score)
    pass


@memoize
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


atexit.register(exit_handler)

if __name__ == "__main__":
    fibonacci(15)
