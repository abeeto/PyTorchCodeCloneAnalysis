import torch, sys, inspect, json, logging, argparse
from types import * 
from pathlib import Path

base_path = Path(__file__).parent
if base_path.__str__() not in sys.path:
    sys.path.append(base_path.__str__())

from Color import Color as C
from BluePrint import BluePrint, BluePrintEncoder, BluePrintDecoder
from Utils import class_from_string, str2bool, valid_dir_path, parse_key_value_pairs
from log import setup_basic_logger

class HyperParameters(object):
    dirPath = None

    def __init__(self, id, blueprint, load_existing=False, custom_dir=None, log_level=logging.INFO, **kwargs):
        self.id = id
        self.logger = setup_basic_logger(custom_dir, log_level, f"hyperparameters_{id}")
        if HyperParameters.dirPath is None:
            HyperParameters.dirPath = Path.cwd() / "HyperParameters" if custom_dir is None else custom_dir
            HyperParameters.dirPath.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
            self.logger.info(f"HyperParameters directory at: {HyperParameters.dirPath}")
        if load_existing:
            load_dir = HyperParameters.dirPath if custom_dir is None else custom_dir
            self.load(load_dir)
            self.logger.info(f"Loaded existing Hyperparameters combination at: {load_dir}")
        self.blueprint = blueprint # if isinstance(blueprint, BluePrint) else BluePrint(id=blueprint).load(dirPath=dirPath.parent / "Blueprints")

        for k,v in kwargs.items():
            if k not in self.blueprint:
                self.logger.warning(f"Keyword '{k}' not in BluePrint '{blueprint.id}'")
                continue
            if self.blueprint.check(k, v):
                vars(self).update({k:v})
            else:
                self.logger.warning(f"Keyword '{k}' did not pass BluePrint '{blueprint.id}' constraints. Value: {v}")

    def save(self, custom_dir=None):
        file_path = Path(HyperParameters.dirPath if custom_dir is None else custom_dir) / f"HyperParameters_{self.id}.json"
        with open(file_path, "w") as f:
             json.dump(self, f, cls=HyperParametersEncoder)

    def load(self, custom_dir=None):
        file_path = Path(HyperParameters.dirPath if custom_dir is None else custom_dir) / f"HyperParameters_{self.id}.json"
        if not file_path.exists() and file_path.is_file(): 
            self.logger.error(msg := f"{file_path} does not exist so it could not be loaded into Hyperparameters '{self.id}'.")
            raise FileNotFoundError(msg)
        with open(file_path, "r") as f:
            kwargs = json.load(f, cls=HyperParametersDecoder)
        vars(self).update(kwargs)
        return self

    def fetch_class_hyperparameters(self, cls):
        if not inspect.isclass(cls):
            self.logger.error(msg := f"Attempted to fetch hyper-parameters for non-class: {cls}")
            raise TypeError(msg)
        kwargs = {}
        sig = inspect.signature(cls)
        for param in sig.parameters.values():
            if param.default not in [inspect.Parameter.empty, inspect._empty]:
                # Use existing hyper-parameter as default argument, otherwise use existing default argument
                kwargs.update({param.name: vars(self).get(param.name, param.default)})
        return kwargs

    def wizard(self):
        for key, constraint in vars(self.blueprint).items():
            if key in ["id", "skip_prompts", "logger"]: continue
            is_typed = True if type(constraint) is type else False
            if is_typed: print(f"Enter value of Type {constraint.__name__}")
            else: print("\n".join([f"{i}: {c}" for i, c in enumerate(constraint)]))

            while (resp := input(f"Choose a value for {key}, or ENTER to skip").strip()) != "":
                if is_typed:
                    try:
                        value = constraint(resp)
                        break
                    except:
                        print(f"Response {resp} did not satisfy {key}'s {constraint} constraint")
                else:
                    if resp.isdigit():
                        value = constraint[int(resp)]
                        break
                    else:
                        print(f"Please select a one of the options for {key} by number.")
            if resp != "":
                vars(self).update({key:value})
   
    def get(self, key, default=None):
        item = vars(self).get(key, default) 
         # If hyper-parameter shows up as option for class, use it instead of default
        return (item, self.fetch_class_hyperparameters(item)) if inspect.isclass(item) else item

    def __getitem__(self, key):
        try:
            item = vars(self).get(key)
        except:
            self.logger.error(msg := f"'{key}' not in Hyperparameters '{self.id}'")
            raise KeyError(msg)
         # If hyper-parameter shows up as option for class, use it instead of default
        return (item, self.fetch_class_hyperparameters(item)) if inspect.isclass(item) else item

    def __setitem__(self, key, value, update_blueprint=False):
        if update_blueprint:
            try:
                self.blueprint[key] = value
            except (ValueError, TypeError):
                self.logger.warning(msg := f"Failed to update Blueprint '{self.blueprint.id}' with key '{key}' and value: {value}")
        if key in self.blueprint:
            if self.blueprint.check(key, value):
                vars(self).update({key:value})
            else:
                self.logger.error(msg := f"Value {value} for key {key} does not meet constraint: {self.blueprint[key]}")
                raise ValueError(msg)
        else: # Hyperparameter only exists in this instance
            vars(self).update({key:value})

    def __eq__(self, other):
        x, y = vars(self), vars(other)
        for k in {**x, **y}.keys():
            if k == "id": continue
            if k not in x or k not in y: return False
            if x[k] != y[k]: return False
        return True

    def __str__(self):
        info_str = f">>> {C.BOLD} Hyper-Parameters '{self.id}' {C.END} <<<"
        for key,value in vars(self).items():
            if key in [*inspect.signature(self.__init__).parameters.keys(), "logger"]: continue # Ignore __init__ signature params
            if type(value) == tuple:
                info_str += "\n{C.BOLD}{key}{C.END} = {value}"
                if len(value) == 2:
                    if value[1] is not None: info_str += f"\n\t**kwargs = {value[1]}"
            else:
                info_str += f"\n{key} = {value}"
        return info_str

class HyperParametersEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, HyperParameters):
            template = {"id": "default", "__classes__":[],
                        "__primitives__":[]}
            for k, v in vars(obj).items():
                if k == "logger": continue
                if k == "id":
                    template["id"] = v
                elif callable(v):
                    template["__classes__"].append((k,f"{v.__module__}.{v.__name__}"))
                elif type(v) in [str, int, bool, float]:
                    template["__primitives__"].append((k,v))
            return template
        return json.JSONEncoder.default(self, obj)

class HyperParametersDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        
    def object_hook(self, dct):
        if "__classes__" in dct and "__primitives__" in dct: # Is Hyperparameters instance
            kwargs = {}
            for k, v in dct["__classes__"]:
                kwargs[k] = class_from_string(v)
            for k, v in dct["__primitives__"]:
                kwargs[k] = v
            return kwargs
        return dct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='>> Generate a Hyper-Parameter combination using an existing BluePrint <<')
    parser.add_argument('-i', dest="id", type=str, metavar='STR', nargs='?', default="default", const=True,
                        help='Identifier for the Hyper-Parameter combination, will overwrite existing if not unique.')
    parser.add_argument('-b', dest="blueprint_id", type=str, metavar='STR', nargs='?', default="default", const=True,
                    help='Identifier for the previously-created BluePrint, used to check Hyper-parameter values against constraints.')
    parser.add_argument('-l', dest='load_existing', type=str2bool, metavar='BOOL', nargs="?", default=False, const=True,
                        help='Whether to try and load an existing Hyper-Parameter combination with the same id')
    parser.add_argument('-p', dest="custom_dir", type=valid_dir_path, metavar='STR', nargs='?', default=Path.cwd() / "BluePrints", const=True,
                            help='Path to directory containing Hyper-Parameter combinations.')
    parser.add_argument("--set",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="""Set a number of key-value pairs (NO space before/after = sign).
                             If a value contains spaces, you should define it with double quotes:
                             'foo="this is a sentence". Note that values are always treated as strings.""")
    args = parser.parse_args()
    if args.set is None: # Show examples
        kwargs = parse_key_value_pairs(args.set, no_epochs=20, batch_size=10, shuffle="True")
    else:
        kwargs = parse_key_value_pairs(args.set)
    bp = BluePrint(args.blueprint_id)
    bp.load(args.custom_dir)

    hparams = HyperParameters(args.id, blueprint=bp)
    hparams.wizard()
    HyperParameters.save(hparams)
    hparams2 = HyperParameters(args.id, blueprint=bp)
    hparams2.load()
    print(hparams)
    print(hparams2)
    print(f"Initial HParams == Reloaded HParams: {hparams == hparams2}")