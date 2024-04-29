def class_from_string(s):
    """
    Flexibly evaluates a string qualifier to allow for non-standard package hierarchies.
    Input: 
        Full object qualifier (str). Composed of module followed by class name 'MODULE_NAME.CLASS_NAME'
    Output:
        Class
    """
    try:
        cls = eval(s) # E.g. torch.nn.modules.loss.BSEloss
        return cls
    except:
        pass
    parts = s.split(".")
     # E.g. torch.optim.asgd.ASGD -> torch.optim.ASGD
    exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}") # Try to import appropriate module
    return eval(parts[-1]) 


def str2bool(v):
    """
    Converts String to boolean, with error handling.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def valid_dir_path(string):
    from pathlib import Path
    if Path(string).is_dir():
        return string
    else:
        raise NotADirectoryError(string)


def parse_key_value_pairs(items, **kwargs):
    """
    Parse (key, value) pairs, seperated by '='. 
    Enforces some eval safety by strict regex matching.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    Credit: @fralau (Aug 25 2018), modified by @deuce1957 (Jan 16 2021)
    """
    import re
    d = {}
    if items is None: return kwargs
    for item in items:
        key, value = item.split('=')
        key = key.strip() # we remove blanks around keys, as is logical
        if re.match("^int|bool|str|float$", value): # Primitive
            d[key] = eval(value)
        elif re.match("^(\w+[\.]*)+$", value): # Module
            d[key] = class_from_string(value)
        elif re.match("^\[(([\w+\.]+)\,*\s*)+\]$", value): # List of classes
            d[key] = [class_from_string(elt) for elt in value.strip("[]").split(",")]
        else:
            raise ValueError(f"Value {value} for key {key} is not a primitive, module or list of classes.")
    return d