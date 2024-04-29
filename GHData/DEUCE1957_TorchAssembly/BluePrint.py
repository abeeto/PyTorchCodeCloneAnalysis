import torch, sys, logging, re, inspect, json, argparse
from types import * 
from pathlib import Path

base_path = Path(__file__).parent
if base_path.__str__() not in sys.path:
    sys.path.append(base_path.__str__())

from Color import Color as C
from Utils import class_from_string, str2bool, valid_dir_path, parse_key_value_pairs
from log import setup_basic_logger

class BluePrint(object):
    dirPath = None

    def __init__(self, id, load_existing=False, custom_dir=None, skip_prompts=False, log_level=logging.INFO, **kwargs):
        self.id = id
        self.skip_prompts = skip_prompts
        self.logger = setup_basic_logger(custom_dir, log_level, f"blueprints_{id}.log")
        if BluePrint.dirPath is None:
            BluePrint.dirPath = Path.cwd() / "BluePrints" if custom_dir is None else custom_dir
            BluePrint.dirPath.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
            self.logger.info(f"BluePrints directory at: {BluePrint.dirPath}")
        if load_existing:
            load_dir = BluePrint.dirPath if custom_dir is None else custom_dir
            self.load(load_dir)
            self.logger.info(f"Loaded existing Blueprint at: {load_dir}")
        for k, v in kwargs.items():
            self._update_variables(k,v) 
    
    def _update_variables(self, k, v):
        if isinstance(v, ModuleType):
            vars(self).update({k:self.get_module_classes(v)})
        elif type(v) is type:
            vars(self).update({k:v})
        elif type(v) is list:
            for elt in v:
                if not inspect.isclass(elt):
                    self.logger.error(msg := f"Element '{v}' in list '{k}' is not a class")
                    raise ValueError(msg)
            vars(self).update({k:v})
        else:
            self.logger.error(msg := f"Value '{v}' for key '{k}' has type {type(v)}, should be Module, List or Type.")
            raise TypeError(msg)
    
    def save(self, dirPath):
        with open(dirPath / f"BluePrint_{self.id}.json", "w") as f:
             print(dirPath / f"BluePrint_{self.id}.json")
             json.dump(self, f, cls=BluePrintEncoder)
        return self

    def load(self, dirPath):
        with open(dirPath / f"BluePrint_{self.id}.json", "r") as f:#
            kwargs = json.load(f, cls=BluePrintDecoder)
        vars(self).update(kwargs)
        return self

    def get_module_classes(self, module):
        cls_dict = {}
        for count, (name, cls) in enumerate(inspect.getmembers(module, inspect.isclass)):
            if re.search("__", name) or re.search("^_|_$", name):
                continue
            desc = cls.__doc__
            desc = "No Description Found" if desc is None else desc.split("\n")[0]
            if not self.skip_prompts:
                print(f"{count}: {C.BOLD}{name}{C.END} ({desc})")
            cls_dict[count] = cls
        resp = "x" if self.skip_prompts else input("Provide digit (e.g. 3), or list of digits (e.g. 2,4,6), of elements to IGNORE. 'x' to exit")
        while resp != "x":
            numbers = resp.split(",")
            for number in numbers:
                if number.isdigit():
                    try:
                        del cls_dict[int(number)]
                    except:
                        print(f"Could not remove '{number}' from elements")
                else:
                    print(f"{number} is not a digit")
            print("--------------")
            for count, cls in cls_dict.items():
                desc = cls.__doc__
                desc = "No Description Found" if desc is None else  desc.split("\n")[0]
                print(f"{count}: {C.BOLD}{cls.__name__}{C.END} ({desc})")
            resp = input("Provide digit or list of digits to ignore, 'x' to exit")
        return [class_from_string(f"{module.__name__}.{cls.__name__}") for cls in cls_dict.values()]

    def check(self, key, item):
        if key not in self:
            self.logger.error(msg := f"'{key}' not in BluePrint '{self.id}'")
            raise KeyError(msg)
        constraint = self[key]
        if type(constraint) is list:
            return True if item in constraint else False
        elif type(constraint) is type:
            return True if type(item) is constraint else False
        else:
            return True if item == constraint else False

    def __getitem__(self, key):
        try:
            item = vars(self).get(key)
        except:
            self.logger.error(msg := f"'{key}' not in BluePrint '{self.id}'")
            raise KeyError(msg)
        return item

    def __setitem__(self, key, value):
        self._update_variables(key,value) 

    def __contains__(self, item):
        return True if item in vars(self) else False

    def __eq__(self, other):
        x, y = vars(self), vars(other)
        for k in {**x, **y}.keys():
            if k == "id": continue
            if k not in x or k not in y: return False
            if x[k] != y[k]: return False
        return True

    def __str__(self):
        info_str = f">>> {C.BOLD}BluePrint {self.id}{C.END} <<<"
        for key, value in vars(self).items():
            if key in inspect.signature(self.__init__).parameters.keys(): continue  # Ignore __init__ signature params
            info_str += f"\n{C.BOLD}{key}{C.END}={value.__name__ if isinstance(value, (ModuleType, type)) else value}" # ToDo: Pretty Print 
        return info_str

class BluePrintEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, BluePrint):
            template = {"id": "default", "__modules__":[],
                        "__types__":[], "__values__":[], "__lists__":[]}
            for k, v in vars(obj).items():
                if k == "id":
                    template["id"] = v
                elif isinstance(v, ModuleType):
                    template["__modules__"].append((k, v.__name__))
                elif isinstance(v, type):
                    template["__types__"].append((k,v.__name__))
                elif type(v) in [str, int, bool, float]:
                    template["__values__"].append((k,v))
                elif type(v) is list:
                    template["__lists__"].append((k, [f"{m.__module__}.{m.__name__}" for m in v]))
            return template
        return json.JSONEncoder.default(self, obj)

class BluePrintDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "__modules__" in dct: # Is BluePrint
            kwargs = {}
            for k, v in dct["__modules__"] + dct["__types__"]:
                # ToDo: Add Import Statement + Error Handling
                kwargs[k] = eval(v)
            for k, v in dct["__values__"]:
                kwargs[k] = v
            for k, v in dct["__lists__"]:
                kwargs[k] = [class_from_string(m) for m in v]
            return kwargs
        return dct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='>> Generate a Blueprint for future hyper-parameter combinations <<')
    parser.add_argument('-i', dest="id", type=str, metavar='STR', nargs='?', default="default", const=True,
                        help='Identifier for the BluePrint, will overwrite existing if not unique.')
    parser.add_argument('-l', dest='load_existing', type=str2bool, metavar='BOOL', nargs="?", default=False, const=True,
                        help='Whether to try and load an existing Blueprint with the same id')
    parser.add_argument('-p', dest="custom_dir", type=valid_dir_path, metavar='STR', nargs='?', default=Path.cwd() / "BluePrints", const=True,
                            help='Path to directory containing Blueprints.')
    parser.add_argument('--log', dest="log_level", type=str, metavar="STR", nargs='?', default="INFO",
                            help="Logging level (Choose: CRITICAL, ERROR, WARNING, INFO, DEBUG")
    parser.add_argument("--set",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="""Set a number of key-value pairs (NO space before/after = sign).
                             If a value contains spaces, you should define it with double quotes:
                             'foo="this is a sentence". Note that values are always treated as strings.""")
    args = parser.parse_args()
    # Additional params here are fallbacks if no args.set provided
    if args.set is None: # Show examples
        print("No key-value pairs provided, showing examples.")
        kwargs = parse_key_value_pairs(args.set, no_epochs=int, epsilon=float, shuffle=bool, optimizer=torch.optim)
    else:
        kwargs = parse_key_value_pairs(args.set)
         
    bp = BluePrint(args.id, args.load_existing, args.custom_dir, log_level=getattr(logging, args.log_level),
        **kwargs,
    )
    BluePrint.save(bp, args.custom_dir)
    bp2 = BluePrint(args.id)
    bp2.load(args.custom_dir)
    print(bp)
    print(bp2)
    print(f"\n{C.BOLD}Initial Blueprint{C.END} == {C.BOLD}Reloaded Blueprint{C.END}: {bp == bp2}")