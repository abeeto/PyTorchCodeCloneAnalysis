from os.path import dirname, basename, isfile, join
import glob
import sys


def import_all_from_directory(directory):
    modules = glob.glob(join(dirname(f"{directory}/"), "*.py"))

    for py in  [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]:
        mod = __import__('.'.join([directory, py]), fromlist=[py])
        classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for cls in classes:
            setattr(sys.modules[directory], cls.__name__, cls)


def create_instance(instance_name, scope, *args, **kwargs):
    try:
        instance = scope[instance_name]
    except ImportError:
        raise ImportError(instance_name)
    except AttributeError:
        raise ImportError(instance_name)
    instance_functor = instance(*args, **kwargs)
    return instance_functor


def get_class(instance_name, scope):
    try:
        instance = scope[instance_name]
    except ImportError:
        raise ImportError(instance_name)
    except AttributeError:
        raise ImportError(instance_name)
    return instance
