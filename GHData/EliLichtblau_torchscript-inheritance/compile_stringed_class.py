import torch
from typing import Dict, Tuple, Type, Any
import inspect
from class_to_string import inline_class
import re
REGEX_GRAB_INIT_METHOD = re.compile(r"(def\s+__init__\(.*\).*:\s*\n(.+\n?)+)(def\s+)?")
REGEX_GRAB_INIT_ARGS = re.compile(r"def\s+__init__\(self\s*,(.*)\)")

#                  Dict[python_class_type] -> ( Dict[(init_type_definitions)] -> Tuple[jitted_class, class_string_repr_python]
GLOBAL_CLASS_DICT: Dict[Type, Dict[Tuple[Any,...], Tuple[Any, str]]] = dict()
#                  Dict[python_class_type] -> (arg_types, str_representation)
COMPILED_DEFAULTS: Dict[Type, Tuple[Any, str]] = dict() # just stores the default compilation without any madness... used as base to create other instances
from class_to_string import get_function_parameters

some_global_var = 10

def get_class_default_types(class_type: Type) -> Tuple[Tuple[Type,...], Dict[str, Type]]:
    """
    Parameter class_type is a Type, just some class, needs to be inspectable
    Returns a tuple of the types of the parameters in the init method
    If there is no type hint we assume a torch tensor so
    I.E
    def __init__(self, x: int, y, z: Tuple[int]):
        ....
    returns (int, torch.Tensor, Tuple[int]), kwarg is a dict of (kwarg_name, type)
    """
    passed_args, passed_kwargs = get_function_parameters(inspect.getsource(class_type.__init__))
    del passed_args["self"] # drop self
    arg_types = tuple(eval(tp) if tp is not None else torch.Tensor for tp in passed_args.values())
    kwarg_types: Dict[str, Type] = dict()
    for str_kwg_name, (str_type, _) in passed_kwargs.items():
        if str_type is None:
            kwarg_types[str_kwg_name] = torch.Tensor
        else:
            kwarg_types[str_kwg_name] = eval(str_type)
    return arg_types, kwarg_types



def _grab_init_method(class_source: str) -> str:
    """
    Expects a string representation of a class and pulls the full __init__ method and returns it as a string
    Yes it has to be a string because it is an inlined_class so just calling inspect getsource will not work
    """
    init_method = re.findall(REGEX_GRAB_INIT_METHOD, class_source)
    # I think this is stupid but I'm going to leave this here for now
    if len(init_method) != 1:
        raise RuntimeError(f"_grab_init_method failure:\nclass source:\n{class_source}\ninit_method captured:\n{init_method}")
    init_method = init_method[0]

    if isinstance(init_method, tuple):
        return init_method[0]
    return init_method

def _compile_default_class(_class: Type) -> bool:
    """
    Param: _class a class Type I.e 
        class Dummy: 
            pass

    If there exists no default compilation for this class, compile it
    and push it to GLOBAL_CLASS_DICT

    Returns False if the class was already compiled, true otherwise
    """


    arg_types_to_str_class_jit: Dict = dict()
    if _class in GLOBAL_CLASS_DICT:
        #print("Already compiled")
        return False

    arg_types, kwarg_types = get_class_default_types(_class)
    # TODO: like execute jit here
    # TODO: handle kwargs
    inlined_stringed_class = inline_class(_class)
    arg_types_to_str_class_jit[arg_types] = tuple((None, inlined_stringed_class))
    GLOBAL_CLASS_DICT[_class] = arg_types_to_str_class_jit
    COMPILED_DEFAULTS[_class] = tuple((arg_types, inlined_stringed_class))
    #breakpoint()
    return True


def _grab_type(arg_or_kwarg: str) -> str:
    if "=" in arg_or_kwarg:
        name_type, val = arg_or_kwarg.split("=")
        if ":" in name_type:
            name, _type = name_type.split(":")
            return _type
        return "torch.Tensor"
    if ":" in arg_or_kwarg:
        name, _type = arg_or_kwarg.split(":")
        return _type
    return "torch.Tensor"

def _replace_init_parameters(_class: Type, new_arg_types: Tuple[Type,...]) -> str:
    """
    Parameters: _class which is a python a type
    new_arg_types and the types passed to the object instance
    in the class we are going to have to recompile
    """
    (default_arg_types, default_compilation_string) = COMPILED_DEFAULTS[_class]
    init_method = _grab_init_method(default_compilation_string)
    old_args = re.findall(REGEX_GRAB_INIT_ARGS, init_method)
    old_args: str = old_args[0]
    split_old_args = old_args.split(",")
    split_new_args = []
    #breakpoint()
    # TODO: assert is subtype
    for i,old_arg in enumerate(split_old_args):
        if "=" in old_arg:
            break # ya this is stupid
        #breakpoint()
        old_type = _grab_type(old_arg)
        new_arg = old_arg.replace(old_type, str(new_arg_types[i]))
        split_new_args.append(new_arg)
    new_args = ''.join(split_new_args)

    new_init_method = init_method.replace(old_args, new_args)
    return new_init_method


def look_for_type_scheme_compile_if_not_found(_class: Type, *args, **kwargs):
    """
    Takes a class and tries to find a compiled version with its type signature,
    If it can't find anything it will compile with the new type signature
    """
    # TODO: add kwarg implementation parameters
    
    # currently just works for args
    if _class not in GLOBAL_CLASS_DICT:
        raise RuntimeError(f"{_class} has no compiled default... this function is not meant to do that")
    
    # get passed arg types
    passed_arg_types = tuple(type(arg) for arg in args)
    compiled_classes = GLOBAL_CLASS_DICT[_class]
    #breakpoint()
    if passed_arg_types not in compiled_classes:
        # TODO: Also need to recompile functions in class? do i want to store these differently some how? 

        #arg_types, _ = get_class_default_types(_class)
        # I believe we can pass down type checking to prebuilt torchscript compiler which is nice
        # so basically when something does not match it is passed down the chain
        (default_arg_types, default_compilation_string) = COMPILED_DEFAULTS[_class]

        
        init_method = _grab_init_method(default_compilation_string)
        #args_to_type, _ = get_function_parameters(init_method)
        # replace type specifiers in init with passed args
        # TODO: conditionionally recompile functions used in init 
        # i.e given def __init__(..., x: oldType): some_var = compiled_function(x)
        # if we now have def __init__(..., x: newType): -> we need to recompile the compiled function

        new_init_method = _replace_init_parameters(_class, passed_arg_types)
        new_class_instance = default_compilation_string.replace(init_method, new_init_method)
        # push new compiled representation into global class dict
        new_repr = tuple((None, new_class_instance))
        GLOBAL_CLASS_DICT[_class][passed_arg_types] = new_repr

        


        


def jit(_class):
    def decorator(*args, **kwargs):
        print(args, kwargs)
        
        _compile_default_class(_class) # compile the class
        # look up type signature
        #breakpoint()
        # what the fuck python am i jsut stupid about
        if kwargs is not {}:
            look_for_type_scheme_compile_if_not_found(_class, *args, **kwargs)
        else:
            look_for_type_scheme_compile_if_not_found(_class, *args)

        #breakpoint()
        return _class(args, kwargs)
    
    return decorator


@jit
class Test:
    def __init__(self, x: int, intTup: Tuple[int], kwg: int =10, utp = 30):
        self.x = x
        self.var2 = 10
        self.var3 = 20


t = Test(5,4)

#t = torch.jit.script(Test(10))
#print(type(t))
"""
@torch.jit.script
def square(x: int):
    return x*x

@torch.jit.script
class Test2:
    def __init__(self, x: torch.Tensor):
        self.x = square(x)
    
    def method1(self, x):
        return x*x*x

"""


breakpoint()
#get_class_default_types(Test)

#t = Test(5, kwg=40)


if __name__ == "__main__":
    pass