"""

Preprocessing decorators for torchscript. These will define
interactions related to scripting allowing feats such as
inline function declaration and class based inheritence.

This is in turn made possible by recompiling an individual
unit of code and extracting or rewriting the code to be
compatible with the standard torchscript parser.

"""
import inspect
from typing import Callable

#### Inline Function ####
import astroid




class inline_function():
    """
    The inline decorator. Declares that the
    following block of code is an inline function
    when evaluated.

    This does not bypass restrictions on pass
    types torchscript enforces

    design
    ######

    In practice, the decorator has the effect of
    compiling a class which displays the top level
    class equivalent, and also indicates to the
    preprocessor when preprocessed that this needs
    to be rewritten. In the general case, of N different
    used context variables, and then M different call parameters

    ::
        var_1: T1 = ...
        var_2: T2 = ...
        ...
        var_N: TN = ...

        @inline_function
        def function(param_1: TT1, ... param_m: TTM):
            docstring
            ...

    This will result in a class encapsulating a compiled format
    as

    ::
        class __{qualified_name}_function():
            def __init__(self, var1: T1, ... varN: TN):
                self.var1 = var1
                self.var2 = var2
                ...
                self.varn = varN
            def __call__(self, param1: TT1, ...):
    As an example, a compiled version of the following function

    ::
         def function():
           item = 5
           @inline_function
           def internal(item2: int):
              return item, item2
           return internal

    Would look like this:

    ::
        class {}_internal():
            def __init__(self, item: int):
                self.item = item
            def callable(item2: int, item: int):


    """
    def __init__(self, function: Callable):
        """
        Initialized with the function to convert.

        Conversion happens at compiletime, due to
        the need to have a tree in which to infer
        types on.

        :param function:
        """
        assert inspect.isfunction(function)
        self.function = function
    def compile(self, node: astroid.FunctionDef):








def inline_function()