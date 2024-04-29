import sys
import inspect


# SYSTEM TOOLS
def is_local_function(obj):
    return callable(obj) and obj.__module__ == __name__


def get_local_functions(return_dict=True):
    local_functions = inspect.getmembers(sys.modules[__name__], predicate=is_local_function)
    if return_dict:
        return dict(k=v for k, v in local_functions)
    return [k for k, v in local_functions]


# FUNCTIONAL PROGRAMMING TOOLS

class Maybe:
    def __init__(self, *value):
        self.value = value

    def apply(self, func, *xs, **kws):
        v = None
        if self.value is not None:
            v = func(*self.value, *xs, **kws)
        return Maybe(v)


# print(Maybe(5).apply(lambda x: x + 3))
