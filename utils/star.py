import functools


def star(f):
    @functools.wraps(f)
    def f_inner(args):
        return f(*args)

    return f_inner
