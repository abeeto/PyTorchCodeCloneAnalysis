from typing import Union, List, Type, Any, Tuple, Optional, NamedTuple, Callable, Dict
import inspect
import re

REGEX_SUPER_CALL = re.compile(r"super\(.*\)\.__init__\(.*\)")
REGEX_GET_SUPER_ARGS = re.compile(r"super\(.*\)\.__init__\((.*)\)")
REGEX_DUNDER_INIT_ARGS = re.compile(r"def\s+__init__\((.*)\)\s*.*:")

REGEX_FUNCTION_ARGS = re.compile(r"def\s+[^\(]+\((.+)\)")
REGEX_FUNCTION_DEF = re.compile(r"(def\s+[^\(]+\(.*\).*:)")

#                  Dict[python_class_type] -> ( Dict[(init_type_definitions)] -> Tuple[jitted_class, class_string_repr_python]
GLOBAL_CLASS_DICT: Dict[Type, Dict[Tuple[Any,...], Tuple[Any, str]]] = dict()

import string
import random

# TODO: ensure uniqueness rather than just like... what are the odds y'know
def id_generator(size=15, chars=string.ascii_uppercase + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))

class NO_DEFAULT:
    pass



def get_function_body(func: str) -> str:
    """
    Expect:
    def dummy(*args, **kwargs):
        body
    returns: body
    """
    function_definitions = re.findall(REGEX_FUNCTION_DEF, func)
    if len(function_definitions) != 1:
        raise ValueError(f"get_function_body does not support nested functions! given {func}")
    string = re.sub(REGEX_FUNCTION_DEF, "", func)
    return string


def get_function_parameters(func: str) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """
    Expects:
    def dummy(*args, **kwargs):
        ...
    returns: dict[*args] = types, dict[*kwargs] = (types, defaults)
    """
    parameters = re.findall(REGEX_FUNCTION_ARGS, func)
    if len(parameters) != 1:
        raise ValueError(f"get_function_arguments does not support nested functions: passed func {func}")
    parameters: str = parameters[0]
    split_params = parameters.split(",")

    arg_to_type: Dict[str, str] = dict()
    kwargs_to_default_and_type: Dict[str, Tuple[str, str]] = dict()
    for param in split_params:
        if "=" in param: #kwarg stuff
            arg_and_type, default = param.split("=")
            if ":" in arg_and_type:
                arg, type_hint = arg_and_type.split()
                kwargs_to_default_and_type[arg.strip()] = (type_hint, default)
            else:
                kwargs_to_default_and_type[arg_and_type] = (None, default)

        else:
            if ":" in param:
                arg, type_hint = param.split(":")
                arg_to_type[arg.strip()] = type_hint.strip()
            else:
                arg_to_type[param.strip()] = None

    return arg_to_type, kwargs_to_default_and_type

def get_super_call(_init_method: str) -> str:
    super_call = re.findall(REGEX_SUPER_CALL, _init_method)
    if len(super_call) != 1:
            raise ValueError(f"Error passed init {_init_method} made multiple calls to super!")
    return super_call[0]

def get_super_passed_args_and_kwargs(super_call: str) -> Tuple[Tuple[str,...],  Tuple[Tuple[str, str], ...]]:
    """
    Expects: super(self, Type).__init__(*args, **kwargs) || super().__init__(*args, **kwargs)
    Returns: (arg1, arg2, ..), ( (kwarg1, default1), (kwarg2, default2)....)
    """
    passed_args = re.findall(REGEX_GET_SUPER_ARGS, super_call)
    if len(passed_args) != 1:
        raise ValueError(f"get_super_passed_args_and_kwargs: expects super().__init__, recieved {super_call}")
    passed_args: str = passed_args[0]
    split_args = passed_args.split(",")
    args: List[str] = list()
    kwargs: List[Tuple[str, str]] = list()
    for arg in split_args:
        if "=" in arg: #iskwarg
            # do kwarg stuff
            split_kwarg = arg.split("=")
            if len(split_kwarg) != 2:
                raise ValueError(f"get_super_passed_args_and_kwargs split kwarg failure {arg}")
            kwargs.append((split_kwarg[0].strip(), split_kwarg[1].strip()))
        else:
            args.append(arg.strip())
    
    return tuple(args), tuple(kwargs)




def get_tabbing(function_source: str) -> str:
    """
    Param: function_body: the body of a function isolated from inspect.getsource
    Returns: a string representing the tabbing for all inserted new code
    Expects spaces, you have to be extra retarded
    I.E 
    class NameSpace:
        def function():
            body
    if passed function would return
    '       '
    """
    
    #return " "*(inspect.indentsize(function_source))
    
    raise RuntimeError("Not implemented fuck me please I hope I never have to implement this")



def push_inherited_methods(sub_class_type: Type, super_class_type: Type, new_class_string: str) -> str:
    """
    class Base:
        ...
    class SubClass(Base):
        ...
    Expects: (SubClass, Base, string)
    where string is the running class we are recreating that can be passed to torchscript
    Returns: updated new_class_string
    """
    base_methods = set(vars(super_class_type).keys()) - set(vars(sub_class_type).keys()) # this is loose could use improved implementation
    #breakpoint()
    base_methods = list(filter(lambda method: not method.startswith("__"), base_methods)) # yes this is jank
    vars_base = vars(super_class_type)
    base_class_indentation = inspect.indentsize(new_class_string)
    for method_name in base_methods:
        method = vars_base[method_name]
        method_source = inspect.getsource(method).strip() + "\n" # I still want the \n, prolly a less stupid way of doing this
        new_class_string += "\n"+(base_class_indentation+4)*" " + method_source
        #breakpoint()
        
    return new_class_string




def inline_class(_class: Type):
    parents = _class.__mro__[1:-1]
    if len(parents) == 0:
        #raise ValueError("Provided class inherits jack shit")
        return inspect.getsource(_class)
    first_parent = parents[0] # only dealing with you for now
    #init_source: Optional[str] = None

    init_source = inspect.getsource(_class.__init__)
    super_source = inspect.getsource(first_parent.__init__)
    super_call = get_super_call(init_source)

    super_passed_args, super_passed_kwargs  = get_super_passed_args_and_kwargs(super_call)
    super_def_args, super_def_kwargs = get_function_parameters(super_source)
    cp = super_source
    # replace super code with passed_args
    super_body = get_function_body(super_source)

    del super_def_args["self"] # ya know shouldn't be there lel
    if len(super_passed_args) != len(super_def_args):
        raise ValueError(f"Super expected {len(super_def_args)} args, recieved {len(super_passed_args)} ")

    # replace passed args
    for passed_arg, (def_arg, def_type) in zip(super_passed_args, super_def_args.items()):
        # Should do some type check stuff prolly
        super_body = super_body.replace(def_arg, passed_arg)
    

    # replace pass kwargs
    for (passed_kwarg_name, passed_kwarg_value), (def_kwarg, (def_type, def_default)) in zip(super_passed_kwargs, super_def_kwargs.items()):
        super_body = super_body.replace(def_kwarg, passed_kwarg_name)

    # replace non replaced default kwargs with their default values
    define_kwargs = []
    for def_kwarg, (def_type, def_default) in super_def_kwargs.items():
        define_kwargs.append(f"{def_kwarg.strip()} {def_type} = {def_default}\n")
    
    kwargs_def_string = ''.join(define_kwargs)
    super_body = kwargs_def_string + super_body


    old_super_body = get_function_body(super_source)
    new_init_source = init_source.replace(super_call, super_body)
    
    class_source = inspect.getsource(_class)
    new_class_source = class_source.replace(init_source, new_init_source)



    # -------- copy methods defined in super_class -------
    new_class_source = push_inherited_methods(_class, first_parent, new_class_source)

    # replace class name
    new_name = id_generator()
    new_class_source = re.sub(f"class\\s+{_class.__name__}"+r"(\(.*\))", f"class {new_name}", new_class_source)
    return new_class_source

