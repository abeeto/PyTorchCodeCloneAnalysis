import ast
import inspect
import pathlib
import warnings
import string
import torch
import sys
from collections import namedtuple
from torch import _sources
from torch.jit import _state
from torch.jit import annotations
from torch._C._jit_tree_views import (
    ClassDef, Ident, Stmt, Decl, Def, Var,
    EmptyTypeAnnotation, Param, ExprStmt, Assign,
    Delete, Return, Raise, Assert, AugAssign, While,
    For, If, Pass, Break, Continue, Apply, Dots, Select,
    TrueLiteral, FalseLiteral, NoneLiteral, Starred,
    ListLiteral, TupleLiteral, DictLiteral, Const,
    StringLiteral, ListComp, Attribute, BinOp, UnaryOp,
    SliceExpr, Subscript, TernaryIf, With, WithItem, Property,
    DictComp,
)
from torch import _sources

# Typing control. For sanity purposes,
# everything within this file are strongly typed.
#
# This means, however, we are sometimes facing some rather
# complex types. They are defined in the code below

from typing import Tuple, Union, List, Optional, Dict, Any, Callable, Collection

# TYPING CONTROL
#These are the types the C++ backend make available, and
#the kind of nodes the compiler should be returning
CNode = Union[
    ClassDef, Ident, Stmt, Decl, Def, Var,
    EmptyTypeAnnotation, Param, ExprStmt, Assign,
    Delete, Return, Raise, Assert, AugAssign, While,
    For, If, Pass, Break, Continue, Apply, Dots, Select,
    TrueLiteral, FalseLiteral, NoneLiteral, Starred,
    ListLiteral, TupleLiteral, DictLiteral, Const,
    StringLiteral, ListComp, Attribute, BinOp, UnaryOp,
    SliceExpr, Subscript, TernaryIf, With, WithItem, Property,
    DictComp,
]

# Reservations begin. Anything located here is being reserved for the scripting function

_reserved_prefix = '__jit'
_reserved_names = {'print'}
_identifier_chars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)

## PLACEHOLDERS ###



#Notes:

# The Print Stmt statement is unreachable
# The ignore context handler still needs to be more robust
# The "is_ignore_context" checker appears to have extra lines?
# The "is_ignore_context" checker has a bug:
#   if torch is not imported as torch, the checker will fail to false when it should be true.
# It is not clear what the multiple entries in an astroid.With.items field are for. Inside it is a (Context, Assign) Tuple.
#   How can you initialize multiple context wiht the one statement?
# "self" is hardcoded as the self variable. Probably a bad idea.
# I am not sure what the NamedConstant ast node does. It is not documented
# There is a bug in the error handling code for build_Unop in torch
# When the "Apply" CNode is created, it goes to c++... which then somehow

#Pretty error handling dictionaries. The following
#features have NO relevance to parsing, but just
#let the error messages come out looking nice.

pretty_node_names = {
    astroid.FunctionDef: "function definitions",
    astroid.For: "for loops",
    astroid.Delete: "del statements",
    astroid.ClassDef: "class definitions",
    astroid.With: "with statements",
    astroid.Raise: "raise statements",
    astroid.Assert: "assertions",
    astroid.Import: "import statements",
    astroid.ImportFrom: "import statements",
    astroid.Global: "global variables",
    astroid.Break: "break statements",
    astroid.Continue: "continue statements",
}

node_start_tokens = {
    astroid.FunctionDef: "def",
    astroid.For: "for",
    astroid.Delete: "del",
    astroid.ClassDef: "class",
    astroid.With: "with",
    astroid.Raise: "raise",
    astroid.Assert: "assert",
    astroid.Import: "import",
    astroid.ImportFrom: "from",
    astroid.Global: "global",
    astroid.Break: "break",
    astroid.Continue: "continue",
}

pretty_node_names.update({
    astroid.AsyncFunctionDef: "async function definitions",
    astroid.AsyncFor: "async for loops",
    astroid.AsyncWith: "async with statements",
    astroid.TryFinally: "try finally blocks",
    astroid.TryExcept: "try except blocks"
    astroid.Nonlocal: "nonlocal variables",
})

node_start_tokens.update({
    astroid.AsyncFunctionDef: "async def",
    astroid.AsyncFor: "async for",
    astroid.AsyncWith: "async with",
    astroid.Try: "try",
    astroid.Nonlocal: "nonlocal",
})

if sys.version_info >= (3, 6):
    pretty_node_names.update({
        astroid.AnnAssign: "annotated assignments",
    })
    # NB: no specific token for AnnAssign

#Error Handling functions and utilities
def find_before(ctx: _sources.SourceContext, pos: int, substr: str, offsets: Tuple[int, int]=(0, 0)):
    new_pos = ctx.source[:pos].rindex(substr)
    return ctx.make_raw_range(new_pos + offsets[0], new_pos + len(substr) + offsets[1])

class FrontendError(Exception):
    def __init__(self, source_range, msg):
        self.source_range = source_range
        self.msg = msg

        # This has to be instantiated here so the ErrorReport is accurate to the
        # call stack when the FrontendError was raised
        self.error_report = torch._C.ErrorReport(self.source_range)
    def __str__(self):
        return self.msg + self.error_report.what().lstrip()


class NotSupportedError(FrontendError):
    pass

class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason=''):
        # If we don't have a specific token, we default to length of 1
        node_type = type(offending_node)
        range_len = len(node_start_tokens.get(node_type, ' '))
        source_range = ctx.make_range(offending_node.lineno,
                                      offending_node.col_offset,
                                      offending_node.col_offset + range_len)
        feature_name = pretty_node_names.get(node_type, node_type.__name__)
        msg = "{} {}aren't supported".format(feature_name, reason + ' ' if reason else '')
        super(UnsupportedNodeError, self).__init__(source_range, msg)


# Present here are logical "is" check statements. These are compiler support features

def is_reserved_name(name):
    return name.startswith(_reserved_prefix) or name in _reserved_names

def is_torch_jit_ignore_context_manager(node: astroid.With):
    """ This function checks if a function in a with node is opening an ignore context"""
    # TODO: more robust handling of recognizing ignore context manager
    # To be more clear, the ignore_context_manager checker does not in fact
    # currently check if we are making a call to the ignore context function.
    #
    # Rather, it just checks by a series of if statements whether it looks like we are
    # calling a function with the same name. This

    #Check if we are opening the 'with' node with a function at all
    first_node = node.items[0]
    if not isinstance(first_node, astroid.Call):
        return False

    #Extract function. Then check if the call has an attribute section. (Why is this needed???)
    function = first_node.func
    if not isinstance(function, astroid.Attribute):
        return False

    #All tasks at hand have been cleared. Go get the attribute properties and check them

    attr_name = function.attrname
    attr_value = function.expr
    if not (attr_name == "_IgnoreContextManager" and isinstance(attr_value, astroid.Attribute)):
        return False

    if not (attr_value.attrname == "jit" and isinstance(attr_value.expr, astroid.Name)):
        return False

    assert isinstance(attr_value, astroid.Attribute) #Makes IDE happy.
    if not attr_value.expr.name == "torch":
        return False
    return True

_IS_ASTUNPARSE_INSTALLED = False
try:
    import astunparse  # type: ignore[import]
    _IS_ASTUNPARSE_INSTALLED = True
except ImportError:
    pass

#Compiling helper definitions begin. Some functions come out to be extremely long, and thus
#are placed in a separete location

def build_ignore_context_manager(ctx, stmt):

    #This is one hell of a hack.
    #
    #As far as I can tell, what this does is rebuild a scope when
    #torchscript notices it is being accessed using a with statement
    #followed by the _IgnoreContextManager class.
    #
    #The scope is rebuilt into a function with appropriate inputs and
    # outputs, then injected into the global context.
    #
    #The return then becomes a call to the function, with the feature
    #that the function is routed through @torch.jit.ignore first
    #
    #All of this just to avoid using @torch.jit.ignore directly

    #TODO: rebuild this properly once I install astunparse
    InputType = namedtuple('InputType', ['name', 'ann'])
    OutputType = namedtuple('OutputType', ['name', 'ann'])

    def process_ins_outs(args):
        # parse the context manager to figure out inputs and outputs
        # with their annotated types
        # TODO: add input, output validator
        inputs = []
        outputs = []
        for arg in args:
            var_name = arg.arg
            if sys.version_info < (3, 8):
                # Starting python3.8 ast.Str is deprecated
                var_ann = arg.value.s
            else:
                var_ann = arg.value.value
            var_decl_type, var_ann = var_ann.split(":")
            if var_decl_type == "inp":
                inputs.append(InputType(var_name, var_ann))
            if var_decl_type == "out":
                outputs.append(OutputType(var_name, var_ann))
        return inputs, outputs

    def create_unique_name_ext(ctx, stmt):
        # extension will be based on the full path filename plus
        # the line number of original context manager
        fn = re.sub(r'[^a-zA-Z0-9_]', '_', ctx.filename)
        return f"{fn}_{stmt.lineno}"

    def build_return_ann_stmt(outputs):
        return_type_ann = ""
        return_statement_str = "return "
        if len(outputs) == 0:
            return_type_ann += " -> None"
        if len(outputs) == 1:
            return_type_ann = " -> " + outputs[0].ann
            return_statement_str += outputs[0].name
        if len(outputs) > 1:
            return_type_ann = " -> Tuple"
            return_type_ann += "[" + ", ".join([var.ann for var in outputs]) + "]"
            return_statement_str += ", ".join([var.name for var in outputs])
        return return_type_ann, return_statement_str

    def build_args(args):
        return ", ".join([arg.name for arg in args])

    inputs, outputs = process_ins_outs(stmt.items[0].context_expr.keywords)

    # build the replacement function str with given inputs and outputs
    ignore_function_name = "func_ignore_" + create_unique_name_ext(ctx, stmt)
    ignore_function_str = "\ndef " + ignore_function_name
    ignore_function_str += "(" + ", ".join([var.name + " :" + var.ann for var in inputs]) + ")"

    return_ann, return_stmt = build_return_ann_stmt(outputs)
    ignore_function_str += return_ann + ": pass"

    # first create the functionDef object from just declaration
    ignore_function = ast.parse(ignore_function_str).body[0]

    # dump the body of context manager to dummy function
    ignore_function.body = stmt.body  # type: ignore[attr-defined]

    # insert return statement to the function
    return_stmt = ast.parse(return_stmt).body[0]
    ignore_function.body.append(return_stmt)  # type: ignore[attr-defined]

    # registers the custom function in the global context
    ignore_func_str = "@torch.jit.ignore\n" + astunparse.unparse(ignore_function)
    ignore_func_str += "\nglobals()[\"{}\"] = {}".format(ignore_function_name, ignore_function_name)
    exec(ignore_func_str)  # noqa: P204

    # build the statements as:
    # <out_1>, <out_2>, ... = torch.jit.frontend.<func>(<in_1>, <in_2>)
    assign_str_lhs = build_args(outputs)
    # this function will be registered in torch.jit.frontend module by default
    assign_str_rhs = "torch.jit.frontend.{}(".format(ignore_function_name) + build_args(inputs) + ")"

    if len(outputs) > 0:
        assign_str = assign_str_lhs + " = " + assign_str_rhs
    else:
        assign_str = assign_str_rhs
    assign_ast = ast.parse(assign_str).body[0]
    return assign_ast




## Support context generation
#
# Support contexts track the current source code, line numbers, and other
# features utilized. Generally, any time we feel the need to jump lots of
# code, or any time we will need to go compile something independently, we will create
# a new support context.
#


#Static content checking. This runs the typehints.
ContextNodes = (astroid.FunctionDef, astroid.ClassDef)
ContextNodeTypes = Union[ContextNodes]


class TreeManager():
    """

    The tree manager keeps track of the various source documents
    that are open, and provides utilities which can be utilized to
    search for a particular feature, type, or context

    """
    _compile_references = {}#Basically my own version of torch._state

    def __init__(self):
        """
        Initiate a tree from an object.
        :param obj:
        """

        sourcelines, lineno, filename = _sources.get_source_lines_and_file(obj, )
        filepath = inspect.getsourcefile(obj)
        filename = pathlib.Path(filepath).name

        def get_source_lines_and_file(
                obj: Any,
                error_msg: Optional[str] = None,
        ) -> Tuple[List[str], int, Optional[str]]:
            """
            Wrapper around inspect.getsourcelines and inspect.getsourcefile.

            Returns: (sourcelines, file_lino, filename)
            """
            sourcelines, lineno, filename = _sources.get_source_lines_and_file(obj, torch._C.ErrorReport.call_stack()),
class SupportContext(_sources.SourceContext):
    """
    The support context creator. Calling this
    class will create an appropriate support context
    for whatever is needed. It is the case


    A creater of an abstract node for a variety of situations
    """
    #Various status features.
    is_class = False
    is_function = False
    is_method = False
    is_bound = False
    #Mapper tracking
    _node_mapping: Dict[astroid.NodeNG, Any] = {}
    def __init_subclass__(cls, typing: astroid.NodeNG):
        """
        When subclassed, catches the type we are associating
        the context node with. Further context creation
        then routes to that type.

        :param typing: The type of node to register as.
        """
        if typing not in ContextNodes:
            warnings.warn("Node of type %s is not in the static type definitions. Typehints will be inaccurate until this is corrrected")
        cls._node_mapping[typing] = cls

    def __new__(cls, node: ContextNodeTypes):
        """
        This fetches the appropriate node for the input. If there is
        no specialized node, it returns a default construction.

        It is basically specialized to defer to a subclass if available, and
        otherwise create a standard context.
        """
        for typing, output in cls.node_mapping.keys():
            if isinstance(node, typing):
                return output(node)
        return cls(node)

    def __init__(self, node: ContextNodeTypes, uses_true_division: bool = True, funcname: Optional[str] = None):
        """
        :param node: The context node to build from
        :param uses_true_division: ???
        :param funcname: The name of the function
        """

        source = node.as_string()
        filename = pathlib.Path(node.root().file).name
        lineno = node.lineno
        leading_whitespace = node.col_offset
        super().__init__(source, filename, lineno, leading_whitespace, uses_true_division, funcname)
        self.context_root = node


class ClassSupportContext(SupportContext, typing=astroid.ClassDef):
    """
    A class which creates a support context
    for a class from which a variety
    of useful features may be determined.
    """
    def __init__(self, node: astroid.ClassDef):
        #TODO: See if anything is wrong if I provide the class name as the second statement in the super call.
        super().__init__(node, False)
        self.is_class = True

class FunctionSupportContext(SupportContext, typing=astroid.FunctionDef):
    """
    A support context for a function node of some sort.
    Specialized versions of this may have additional attributes

    When intitialized, returns either a FunctionSupportContext,
    MethodSupportContext, or BoundSupportContext for wild
    functions, unbound methods, and bound methods respectively.
    """
    def __new__(cls, node: astroid.FunctionDef):
        """
        The FunctionDef Node to initialize
        """
        if node.is_bound():
            return BoundSupportContext(node)
        elif node.is_method():
            return MethodSupportContext(node)
        elif node.is_function:
            return cls(node)
        else:
            raise ValueError("Node was not named function")

    def __init__(self, node: astroid.FunctionDef):
        self.is_bound = node.is_bound()
        self.is_method = node.is_method()
        self.is_function = node.is_function
        self.function_type = node.type
        super().__init__(node, True, node.name)

class MethodSupportContext(FunctionSupportContext):
    """
    Adds features particularly when dealing with
    methods. Primarily, it adds information on
    who the parent class is.
    """
    def __init__(self, node: astroid.FunctionDef):
        super().__init__(node)
        self.class_node = node.parent

class BoundSupportContext(MethodSupportContext):
    """
    Adds additional features indicating what the
    name of the bound variable is.
    """
    def __init__(self, node: astroid.FunctionDef):
        first_argument = node.args.arguments[0]
        bound_var_name = first_argument.name
        super().__init__(node)
        self.binding_name = bound_var_name

# Block
#
# The block is a simple list which neglects to append anything
# which is "None"
# #

class Block():
    """

    This represents a code block. Generally, this
    will be found anywhere a list would be normally.

    In order to properly handle node creation, nodes
    which returned as 'None' must be stripped. This
    piece of code ensures this happens

    Two options are available. You can either create a
    list as a compilation, in which case all filtering is done
    for you, or can initialize the class and append to it as
    you go, then .dump the contents
    """
    def __init__(self):
        self.body: List[CNode] = []
    def append(self, node: Optional[CNode]):
        if node is not None:
            self.body.append(node)
    def dump(self)-> List[CNode]:
        return self.body
### NodeManagers ###
# The typesearcher is an entity which ooes exactly what it sounds like - searches for
# a particular type. It is handed the current node location,
#
#




### Libraries are listed below.
#
#Libraries contain within them directives
#on how to compile a particular type of node.
#They are designed to be mixed into the
#builder with multiple inheritance.
#
#They are essencially respositories of
#functions, but with the notable feature
#that they can self-call. When this happens,
#it is assumed that the builder would take the
#next step
#
#The builder will ultimately fetch whatever
#function has the same name as the node it is currently
#exposed to.





class AbstractCompileLibMixin():
    """

    The abstract lib simply indicates
    to any IDE's that the class itself
    is callable, and that it then returns
    a CNode

    """
    @staticmethod
    def clean_list(lst: List[Optional[CNode]])->List[CNode]:
        """
        A CNode code block is not valid if it has
        Nones in it. Meanwhile, this is a possible return.

        This strips them out
        :param lst: a list of CNodes
        :return: A stripped list of CNodes
        """
        return [item for item in lst if item is not None]

    @classmethod
    def clean_comprehension(cls, items: Collection, true_branch: Callable = None, condition: Callable = None,
                              false_branch: Callable = None) -> List:
        """

        This handles the task of creating a clean list all by itself. It basically
        replaces a list comprehension, with the notable qualification of stripping
        any list element that is a None

        :param items: The items to compile
        :param condition: A callable. Called on the items to evaluate a condition. Returns a bool
        :param true_branch: Called if the condition returned true
        :param false_branch: Called if the condition returned false
        :return: A list
        """
        if true_branch is not None and false_branch is not None:
            output = [true_branch(item) if condition(item) else false_branch(item) for item in items]
        elif true_branch is not None and condition is not None:
            output = [true_branch(item) for item in items if condition(item)]
        elif true_branch is not None and condition is None:
            output = [true_branch(item) for item in items]

        elif false_branch is not None and condition is not None:
            output = [false_branch(item) for item in items if not condition(item)]
        else:
            raise ValueError("Combination not supported")
        output = [item for item in output if item is not None]
        return output
    def __init__(self):
        super().__init__()
    def __call__(self, ctx: SupportContext, node: astroid.NodeNG)->CNode:
        raise NotImplementedError("This library should not be called directly")


def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None):
    body = py_def.body
    r = ctx.make_range(py_def.lineno + len(py_def.decorator_list),
                       py_def.col_offset,
                       py_def.col_offset + len("def"))

    param_list = build_param_list(ctx, py_def.args, self_name, pdt_arg_types)
    return_type = None
    if getattr(py_def, 'returns', None) is not None:
        return_type = build_expr(ctx, py_def.returns)

    decl = Decl(r, param_list, return_type)
    is_method = self_name is not None
    if type_line is not None:
        type_comment_decl = torch._C.parse_type_comment(type_line)
        decl = torch._C.merge_type_from_type_comment(decl, type_comment_decl, is_method)

    return Def(Ident(r, def_name),
               decl,
               build_stmts(ctx, body))


def build_param_list(ctx, py_args, self_name, pdt_arg_types=None):
    if py_args.kwarg is not None:
        expr = py_args.kwarg
        ctx_range = ctx.make_range(expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg))
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if py_args.vararg is not None:
        expr = py_args.vararg
        ctx_range = ctx.make_range(expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg))
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if len(py_args.kw_defaults) > 0:
        # kw_defaults is a list of the values for the kwargs (which default to None),
        # so they don't actually have line numbers.
        for arg in py_args.kw_defaults:
            if arg is not None:
                ctx_range = build_expr(ctx, arg).range()
                raise NotSupportedError(ctx_range, _vararg_kwarg_err)

    # List of Tuple of args and type as inferred by profile directed typing
    arg_and_types = [(arg, pdt_arg_types[arg.arg] if pdt_arg_types and bool(pdt_arg_types[arg.arg]) else None)
                     for arg in py_args.args]
    arg_and_types_kwonlyargs = [(arg, pdt_arg_types[arg.arg] if pdt_arg_types and bool(pdt_arg_types[arg.arg])
                                else None) for arg in py_args.kwonlyargs]

    result = [build_param(ctx, arg, self_name, kwarg_only=False, pdt_arg_type=arg_type)
              for arg, arg_type in arg_and_types]
    result += [build_param(ctx, arg, self_name, kwarg_only=True, pdt_arg_type=arg_type)
               for arg, arg_type in arg_and_types_kwonlyargs]
    return result

def build_param(ctx, py_arg, self_name, kwarg_only, pdt_arg_type=None):
    # NB: In Python3 py_arg is a pair of (str arg, expr? annotation)
    name = py_arg.arg
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    if getattr(py_arg, 'annotation', None) is not None:
        annotation_expr = build_expr(ctx, py_arg.annotation)
    elif pdt_arg_type:
        annotation_expr = Var(Ident(r, pdt_arg_type))
    elif self_name is not None and name == 'self':
        annotation_expr = Var(Ident(r, self_name))
    else:
        annotation_expr = EmptyTypeAnnotation(r)
    return Param(annotation_expr, Ident(r, name), kwarg_only)


class CompileControlMixin(AbstractCompileLibMixin):
    """
    The compile mixin contains the logic necessary to
    successfully compile diverse statements and situations
    involving classes, functions, and other primary torchscript
    target features.

    It has the effect of sometimes rewriting the ast tree
    when needed in order to support some features that the
    backend would otherwise choke on.

    It replaces much of the logic of the script frontend.
    """
    @staticmethod
    def fetch_argument_context(ctx: FunctionSupportContext, node: astroid.Arguments, name: str):
        """ arguments in astroid are a little more inconvinent. We must build ranges manually"""

        source = node.root().as_string()
        start = node.parent.lineno
        sourcelines = source.splitlines()[start:]
        index = -1
        lineno = 0
        for lineno, line in enumerate(sourcelines):
            index = line.find(name)
            if index is not -1:
                break
        assert index != -1
        context = ctx.make_range(start + lineno, index, index + len(name))
        return context
    @staticmethod
    def parse_subscript(node: astroid.Arguments, node_collection: astroid.NodeNG):
        """Attempts to convert a subscript node back into working types"""


    def build_Argument(self, ctx: FunctionSupportContext, node: astroid.Arguments):
        """Builds an argument node, returning a bunch of parameters"""

        #A significant amount of sanitation is required. In particular
        #anything related to *args or **kwargs is rejected.
        #Todo: Verify error handling functioning correctly
        if node.kwarg is not None:
            #TODO: Impliment a starred keyword case with annotations. Modify build_Call
            ctx_range = self.fetch_argument_context(ctx,node,node.kwarg)
            raise NotSupportedError(ctx_range, "Starred keyword arguments are not yet supported")
        if node.vararg is not None:
            #TODO: Impliment a starred arg with annotation. Modify build_Call
            ctx_range = self.fetch_argument_context(ctx, node, node.vararg)
        if len(node.kw_defaults) > 0:
            # kw_defaults is a list of the values for the kwargs (which default to None),
            # so they don't actually have line numbers.
            for arg in node.kw_defaults:
                if arg is not None:
                    ctx_range = self.fetch_argument_context(ctx, node, arg)
                    raise NotSupportedError(ctx_range, "default keyword arguments for **kwargs not none. This is not supported")

        #Collect annotations as best as is possible.
        annotations = []
        for i, ann in enumerate(node.annotations):
            if ann is not None:
                annotations.append(ann)
                continue
            elif node.type_comment_args is not None and node.type_comment_args[i] is not None:
                annotations.append(node.type_comment_args[i])
                continue
            elif node.parent and hasattr(node.parent, 'type_comment_args') and \
                node.parent.type_comment_args[i] is not None:
                annotations.append(node.parent.type_comment_args[i])
                continue
            else:
                msg = "Could not find all annotations for function on lineno %s. \n Performing dynamic analysis. This may produce incorrect results"
                warnings.warn(msg)



        annotations = node.annotations.copy()
        if node.type_comment_args is not None:
            for i, annotation in enumerate(node)
        if hasattr(node.parent, 'type_comment_args') and node.parent.type_comment_args is not None:



            iterator = zip(node.args, )

        if node.annotations is not None:
            iterator = zip(arg, annotations)



        Params: List[Param] = []


        for arg, annotation in zip(node.args, node.annotations):
            name = arg
            arg = self(ctx, arg)
            r = self.fetch_argument_context(ctx, node, name)
            if annotation is not None:
                annotation = self(ctx, annotation)
            elif hasattr(node.parent, 'type_annotation'):


            name = py_arg.arg
            r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
            if getattr(py_arg, 'annotation', None) is not None:
                annotation_expr = build_expr(ctx, py_arg.annotation)
            elif pdt_arg_type:
                annotation_expr = Var(Ident(r, pdt_arg_type))
            elif self_name is not None and name == 'self':
                annotation_expr = Var(Ident(r, self_name))
            else:
                annotation_expr = EmptyTypeAnnotation(r)
            return Param(annotation_expr, Ident(r, name), kwarg_only)

            arg = self(ctx, arg)
            annotation = self(ctx, annotation)







            node.parent.body.appe
        annotations = node.annotations
        args = node.args
        keywords

        astroid.Name.pytype

    def get_type_line(self, node: astroid.No)->str:
        return annotations.get_type_line()


    support_stack = []
    def build_FunctionDef(self, ctx: SupportContext, node: astroid.Lambda):

        #Break out if already compiled
        already_compiled = _state._try_get_jit_cached_function(node)
        if already_compiled is not None:
            return already_compiled

        #Create compilation context
        new_ctx = SupportContext(node)

        #Go fetch the parameter and return typing. Do not
        #give up until it is found
        parameter_typing = [None]*len(node.args.arguments)

        if node.args.annotations is not None:
            node.


        pass
    def build_FunctionDef(self):
        pass
    def build_ClassDef(self):
        pass

class StmtLibMixin(AbstractCompileLibMixin):
    """

    The statement lib is responsible
    for maintaining the functions which
    are utilized to compile control
    statement, code block compilation,
    and other similar features.

    This is a mixin clas


    """

    #Support fields. These catagorize and indicate the supported features of the parser
    supported_binary_operations = (
        '+',
        '-',
        '*',
        '/',
        '%',
        '|',
        '&',
        '^',
        '<<',
        '>>',
        '**'
    )

    ###
    #
    # Expression statement building logic. The information and qualifications which were originally covered by
    # the StmtBuilder in the frontend class are rebuilt in the following lines.
    #

    def build_AugAssign(self, ctx: SupportContext, node: astroid.AugAssign):
        """ Builds an augment assign node. This is something involving assignment and logic, such as 'var+=3'"""
        lhs = self(ctx, node.target)
        rhs = self(ctx, node.value)
        op = node.op
        if op not in self.supported_binary_operations:
            raise NotSupportedError(
                find_before(ctx, rhs.range().start, '=', offsets=(-1, 0)),
                "unsupported kind of augumented assignment: " + op)
        return AugAssign(lhs, op, rhs)
    def build_Assign(self, ctx: SupportContext, node: astroid.Assign)-> Assign:
        """ builds an assignment node without a type hint"""
        rhs = self(ctx, node.value)
        lhs = self.clean_comprehension(node.targets, lambda: x: self(ctx, x))
        return Assign(lhs, rhs)
    def build_AnnAssign(self, ctx: SupportContext, node: astroid.AnnAssign)->Assign:
        """Builds an assignment involving an annotation"""
        if node.value is None:
            raise UnsupportedNodeError(ctx, node, reason='without assigned value')

        # Disallow type annotations on instance attributes outside of __init__

        if isinstance(node.target, astroid.Attribute) and \
                stmt.target.value.id == "self" and ctx.funcname != "__init__":  # type: ignore[attr-defined]
            start = node.col_offset
            end = node.end_col_offset
            if hasattr(node.annotation, 'id'):
                end += len(f": {node.annotation.id}")
            sr = ctx.make_range(node.lineno, start, end)
            raise ValueError("Type annotations on instance attributes must be declared in "
                             f"__init__, not '{ctx.funcname}': {sr}")

        rhs = self(ctx, node.value)
        lhs = self(ctx, node.target)
        the_type = self(ctx, node.annotation)
        return Assign([lhs], rhs, the_type)

    def build_Delete(self, ctx: SupportContext, node: astroid.Delete)->Delete:
        """Builds a delete node"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        return Delete(r, self.clean_comprehension(node.targets, lambda target: self(ctx, target)))

    def build_Return(self, ctx: SupportContext, node: astroid.Return)->Return:
        """Builds a return node"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        return Return(r, None if node.value is None else self(ctx, node.value))

    def build_Raise(self, ctx: SupportContext, node: astroid.Raise)->Raise:
        """Builds a raise node"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        expr = self(ctx, node.exc)
        return Raise(r, expr)

    def build_Assert(self, ctx: SupportContext, node: astroid.Assert)-> Assert:
        """Builds an assert node"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        test = self(ctx, node.test)
        msg = self(ctx, node.fail) if node.fail is not None else None
        return Assert(r, test, msg)

    def build_While(self, ctx: SupportContext, node: astroid.While)-> While:
        """Builds a while node. Does not support else """
        if node.orelse is not None:
            # TODO: try to recover the location of else:? Python doesn't give us useful
            # annotations in this case
            raise NotSupportedError(None, "else branches of while loops aren't supported")
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)

        condition = self(ctx, node.test)
        body = self.clean_comprehension(node.body, lambda subnode: self(ctx, subnode))
        return While(r, condition, body)

    def build_For(self, ctx: SupportContext, node: astroid.For)-> For:
        """Builds a for node. Does not support else"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        if node.orelse is not None:
            #TODO: try to get the else branch working?
            raise NotSupportedError(r, "else branches of for loops aren't supported")

        i = [self(ctx, node.target)]
        collection = [self(ctx, node.iter)]
        body = self.clean_comprehension(node.body, lambda subnode: self(ctx, subnode))
        return For(r, i, collection, body)

    def build_If(self, ctx: SupportContext, node: astroid.If)-> If:
        """ Builds an if node"""

        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        condition = self(ctx, node.test)
        body = self.clean_comprehension(node.body, lambda subnode: self(ctx, subnode))
        orelse = self.clean_comprehension(node.orelse, lambda subnode: self(ctx, subnode))
        return If(r, condition, body, orelse)


    def build_Print(self, ctx: SupportContext, node: astroid.NodeNG):
        """

        This node is unreachable. No node in the original ast
        tree exists with the name "Print". This means the original
        builder could not look it up.

        It is kept around for historical purposes only.
        """

        #######################################################
        raise NotImplementedError("Print nodes are not supported, and should never arise. Please submit a ticket if it pops up")
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + len("print"))
        if node.dest:
            raise NotSupportedError(r, "print statements with non-default destinations aren't supported")
        args = [self(ctx, val) for val in node.values]
        return ExprStmt(Apply(Var(Ident(r, "print")), args, []))

    def build_Expr(self, ctx: SupportContext, node: astroid.Expr)-> Optional[ExprStmt]:
        """Builds an expression. Notably, this return may in fact be none. If this is the case, the statement
        at the next level will automatically exclude it from the final tree."""
        value = node.value
        if isinstance(value, astroid.Const) and isinstance(value.value, str):
            # If a statement is a string literal expression,
            # then it is a docstring. Just ignore it.
            return None
        else:
            return ExprStmt(self(ctx, value))

    def build_Pass(self, ctx: SupportContext, node: astroid.Pass)-> Pass:
        """A pass node"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        return Pass(r)

    def build_Break(self, ctx: SupportContext, node: astroid.Break)->Break:
        """ Builds a break node"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        return Break(r)

    def build_Continue(self, ctx: SupportContext, node: astroid.Continue)->Continue:
        """Build a continue statement"""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        return Continue(r)

    def build_With(self, ctx: SupportContext, node: astroid.With)-> With:
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        # Handle ignore context manager
        if is_torch_jit_ignore_context_manager(node):
            #TODO: Rebuild the ignore context manager. It is really more of a hack right now.
            #This bears a little bit of additional explanation. A particular noop,
            #_IgnoreContextManager(), may be utilized with a "with statement
            #as in code: with _IgnoreContextManger(): ....
            #
            #When this is done, the content in the code block is ignored, as though
            #passed through a @torch.jit.ignore
            if not _IS_ASTUNPARSE_INSTALLED:
                raise RuntimeError("torch.jit._IgnoreContextManager requires installing Python library `astunparse`,\
                                   please install it in your Python environment")
            assign_ast = build_ignore_context_manager(ctx, node)
            return self(ctx, assign_ast)

        body = self.clean_comprehension(node.body, lambda subnode: self(ctx, subnode))
        items = Block()
        for ContextManager, OptionalNode in node.items:
            #This block compiles the various with
            #conditions which our compiler might be facing.

            #It takes the place of "WithItemBuilder",
            #centralizing code which is not reused.

            lineno = ContextManager.lineno
            start = ContextManager.col_offset
            end = ContextManager.end_col_offset
            r = ctx.make_range(lineno, start, end)
            ContextManager = self(ctx, ContextManager)
            OptionalNode = self(ctx, OptionalNode) if OptionalNode is not None else None
            compiled = WithItem(r, ContextManager, OptionalNode)
            items.append(compiled)
        return With(r, items.dump(), body)
    def __init__(self):
        super().__init__()

class ExprLibMixin(AbstractCompileLibMixin):
    """

    The ExprLibMixin takes on the same role
    that the ExprBuilder originally did. It concerns itself
    primarily with expressions.
    """
    supported_binops = (
        "+",
        "-",
        "*",
        "/",
        "**",
        '//',
        '&',
        '^',
        '|',
        '<<',
        '>>',
        '@'
    )

    supported_unops = (
        'not',
        '-',
        '~'
    )
    supported_boolops = (
        'and',
        'or'
    )
    supported_comparisons = (
        '==',
        '!=',
        '<=',
        '<',
        '>=',
        '>',
        'is',
        'is not',
        'in',
        'not in'
    )

    def build_Attribute(self, ctx: SupportContext, node: astroid.Attribute)->Select:
        """Builds an attribute. This node gets something from an object"""

        #Additional information, namely the end_col_offset, is available
        #in astroid above that in ast. As a resut, we do not have to
        #jump through any weird hoops.
        base = self(ctx, node.expr)
        lineno = node.lineno
        end = node.end_col_offset
        start = end - len(node.attrname)
        name_range = ctx.make_range(lineno, start, end)
        ident = Ident(name_range, node.attrname)
        return Select(base, Ident(name_range, ident))

    def build_Call(self, ctx: SupportContext, node: astroid.Call)->Apply:
        """ Builds a call statement. """
        func = self(ctx, node.func)
        args = self.clean_comprehension(node.args, lambda subnode: self(ctx, subnode))
        for arg in node.starargs:
            output_token = self(ctx, arg)
            args.append(Starred(output_token.range(), output_token))

        kwargs = Block()
        for kw in node.keywords:
            kw_token = self(ctx, kw.value)
            #TODO: Figure out if this is still implimented correctly.
            if not kw.arg:
                raise NotSupportedError(kw_token.range(), 'keyword-arg expansion is not supported')
            kwargs.append(Attribute(Ident(kw_token.range(), kw.arg), kw_token))
        kwargs = kwargs.dump()
        return Apply(func, args, kwargs)
    def build_Ellipsis(self, ctx: SupportContext, node: astroid.Ellipsis)->Dots:
        """ Builds an ellipse statement"""
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + 3)  # len("...") == 3
        return Dots(r)
    def build_Name(self, ctx: SupportContext, node: astroid.Name)->Var:
        """Builds a Name CNode. Reserves a few different prefixes for private torchscript usage."""
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        if node.name.startswith(_reserved_prefix):
            raise NotSupportedError(r, "names of variables used in JIT-ed functions "
                                       "can't start with " + _reserved_prefix)
        if node.name == "True":
            return TrueLiteral(r)
        elif node.name == "False":
            return FalseLiteral(r)
        elif node.name == "None":
            return NoneLiteral(r)
        elif node.name == "Ellipsis":
            return Dots(r)
        return Var(Ident(r, node.name))
    def build_NameConstant(self, ctx: SupportContext, node: astroid.Const):
        """
        This is called only from build_Const.

        Frankly, I have no idea what this is doing exactly.
        """
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        if node.value is True:
            return TrueLiteral(r)
        elif node.value is False:
            return FalseLiteral(r)
        elif node.value is None:
            return NoneLiteral(r)
        elif node.value == Ellipsis:
            return Dots(r)
        else:
            raise ValueError("Name constant value unsupported: " + str(expr.value))
    def build_BinOp(self, ctx: SupportContext, node: astroid.BinOp)->astroid.BinOp:
        """Builds a binary operation, among the operations currently supported"""
        lhs = self(ctx, node.left)
        rhs = self(ctx, node.right)
        op = node.op
        if op not in self.supported_binops:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise NotSupportedError(err_range, "unsupported binary operator: " + op)
        if op == "/" and not ctx.uses_true_division:
            err_range = ctx.make_raw_range(lhs.range(), rhs.range().start)
            raise FrontendError(err_range, 'Division of ints in TorchScript uses Python 3 true '
                                'division semantics. Please put `from __future__ '
                                'import division` at the top of your file')
        return BinOp(op, lhs, rhs)
    def build_UnaryOp(self, ctx: SupportContext, node: astroid.UnaryOp)-> UnaryOp:
        """ Builds a unitory op. This is an op with only one argument"""
        sub_expr = self(ctx, node.operand)
        op = node.op
        if op not in self.supported_unops:
            #TODO: Validate this error logic
            err_range = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
            raise NotSupportedError(err_range, "unsupported unary operator: " + op.__name__)
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + len(op))
        return UnaryOp(r, op, sub_expr)
    def build_BoolOp(self, ctx: SupportContext, node: astroid.BoolOp)-> BinOp:
        """ Builds a bool op. This consists of two bool values, then an operation on them.

            Notably, returns a BinOp as there is no CNode BinOp.
        """
        if len(node.values) < 2:
            raise AssertionError("expected at least 2 values in BoolOp, but got " + str(len(node.values)))
        sub_exprs = self.clean_comprehension(node.values, lambda subnode: self(ctx, subnode))
        op = node.op
        if op not in self.supported_boolops:
            err_range = ctx.make_raw_range(sub_exprs[0].range().end, sub_exprs[1].range().start)
            raise NotSupportedError(err_range, "unsupported boolean operator: " + op.__name__)

        lhs = sub_exprs[0]
        for rhs in sub_exprs[1:]:
            lhs = BinOp(op, lhs, rhs)
        return lhs
    def build_IfExp(self, ctx: SupportContext, node: astroid.IfExp)->TernaryIf:
        """Builds an IfExpr. This is simply an if statement with further else code."""
        return TernaryIf(self(ctx, node.test),
                         self(ctx, node.body),
                         self(ctx, node.orelse))
    def build_Compare(self, ctx: SupportContext, node: astroid.Compare)->Union[BinOp, UnaryOp]:
        """ Build a compare statement. This converts things into a binary, and so perhaps
            unsuprisingly the CNode return is a BinOp or a UnaryOp"""
        lhs = self(ctx, node.left)
        operands = self.clean_comprehension(node.ops, lambda subnode: self(ctx, subnode))
        result = None
        for op, rhs in operands:
            ### The following code is a little complicated. What is happening
            #is that some comparison statements, such as "in", may have
            #many operands being checked on the right hand size.
            #
            #As a result, we check each individual condition and then
            #accumulate these into a final result.

            r = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            if op not in self.supported_comparisons:
                raise NotSupportedError(r, "unsupported comparison operator: " + op.__name__)
            if op == 'not in':
                # NB: `not in` is just `not( in )`, so we don't introduce new tree view
                # but just make it a nested call in our tree view structure
                in_expr = BinOp('in', lhs, rhs)
                cmp_expr = UnaryOp(r, 'not', in_expr)
            else:
                cmp_expr = BinOp(op, lhs, rhs)

            #Accumulate results so far
            if result is None:
                result = cmp_expr
            else:
                result = BinOp('and', result, cmp_expr)
        return result

    def build_SliceExpr(self, ctx: SupportContext, node: astroid.Slice)-> SliceExpr:
        """ Takes a slice node, and builds it into a CNode SliceExpr"""
        lower = self(ctx, node.lower) if node.lower is not None else None
        upper = self(ctx, node.upper) if node.upper is not None else None
        step = self(ctx, node.step) if node.step is not None else None
        r = ctx.make_range(node.lineno, node.col_offset, node.end_col_offset)
        return SliceExpr(r, lower, upper, step)

    def build_Subscript(self, ctx: SupportContext, node: astroid.Subscript)-> Subscript:
        """

        This has been significantly simplified, at the cost of possible backwards compatibility.

        I will need to check how far back in time the various versions of astroid are compatible.
        """
        #TODO: Check astroid compatibility in python < 3.9. Can I install as is, or do I need to
        #include compatibility code?

        base = self(ctx, node.value)

        #Handles modern ast trees
        if isinstance(node.slice, astroid.Tuple):
            nodetuple = node.slice
            indices = self.clean_comprehension(nodetuple.elts, lambda subnode: self(ctx, subnode))
            if len(indices) == 0:
                #Special logic for if we get a length zero tuple, or a tuple typing

                r = ctx.make_range(node.lineno,
                                   node.slice.col_offset,
                                   node.slice.col_offset + 2)
                tup = TupleLiteral(r, [])
                indices.append(tup)
        else:
            indices = [self(ctx, node.slice)]
        return Subscript(base, indices)
    def build_List(self, ctx: SupportContext, node: astroid.List)-> ListLiteral:
        """Builds a list CNode"""


        list_body = self.clean_comprehension(node.elts, lambda subnode: self(ctx, subnode))
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset+1)
        return ListLiteral(r, list_body)
    def build_Tuple(self, ctx: SupportContext, node: astroid.Tuple)-> TupleLiteral:
        """ builds a tuple CNode"""
        tuple_body = self.clean_comprehension(node.elts, lambda subnode: self(ctx, subnode))
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset+1)
        return TupleLiteral(r, tuple_body)
    def build_Dict(self, ctx: SupportContext, node: astroid.Dict)-> DictLiteral:
        range = ctx.make_range(node.lineno, node.col_offset, node.col_offset + 1)
        keys = Block()
        values = Block()
        for item in node.items:
            if isinstance(item, astroid.DictUnpack):
                raise NotSupportedError(range, "Dict expansion (e.g. `{**dict}`) is not supported")
            key, value = item
            keys.append(self(ctx, key))
            values.append(self(ctx, value))
        return DictLiteral(range, keys.dump(), values.dump())
    def build_Constant(self, ctx: SupportContext, node: astroid.Const):
        """

        Builds a constant. This is trickier than it sounds.

        Since torch cares about whether you are working with a int, string, float, etc,
        the compiler must care as well. The CNode backend thus must be compiled appropriately.
        astroid, however, does not give a crap. So we go inspect types to figure out what is going on.

        """


        value = node.value
        if value is None or isinstance(value, bool):
            # NB: this check has to happen before the int check because bool is
            # a subclass of int
            return self.build_NameConstant(ctx, node)
        if isinstance(value, (int, float, complex)):
            #Handles numeric
            return self.build_Num(ctx, node)
        elif isinstance(value, str):
            #Handles string
            return self.build_Str(ctx, node)
        elif isinstance(value, type(Ellipsis)):
            #handles ellipses
            return self.build_Ellipsis(ctx, node)
        else:
            error_range = ctx.make_range(node.lineno, node.col_offset, node.col_offset + len(str(value)))
            raise FrontendError(error_range, "Unknown Constant expression type")
    def build_Num(self, ctx: SupportContext, node: astroid.Const):
        """ Build a numeric quantity. A helper method of build_Const"""

        value = str(node.value)
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + len(value))
        return Const(r, value)
    def build_Str(self, ctx: SupportContext, node: astroid.Const)->StringLiteral:
        """ Build a string Cnode. A helper method of build_Const"""
        value = str(node.value)
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + len(value) + 1)
        return StringLiteral(r, value)
    def build_JoinedStr(self, ctx: SupportContext, node: astroid.JoinedStr)->Apply:
        """Builds a function to join strings together."""
        s = ''
        args = []
        for value in node.values:
            r = ctx.make_range(value.lineno, value.col_offset, value.col_offset + 1)
            # TODO: Verify all this crap works right.
            if isinstance(value, astroid.FormattedValue):
                if value.conversion != -1:
                    raise NotSupportedError(r, 'Don\'t support conversion in JoinedStr')
                if value.format_spec is not None:
                    raise NotSupportedError(r, 'Don\'t support formatting in JoinedStr')
                s += '{}'
                args.append(self(ctx, value.value))
            elif isinstance(value, str):
                s += value
            else:
                raise NotSupportedError(r, 'Unsupported value in JoinedStr')

        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + 1)
        return Apply(Select(StringLiteral(r, s), Ident(r, 'format')), args, [])

    def build_ListComp(self, ctx: SupportContext, node: astroid.ListComp)->ListComp:
        """Build a list comprehension"""
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset)
        if (len(node.generators) != 1):
            raise NotSupportedError(r, "Only a single generator is currently supported")

        if (len(node.generators[0].ifs) != 0):
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        elt_expr = self(ctx, node.elt)
        target_expr = self(ctx, node.generators[0].target)
        iter_expr = self(ctx, node.generators[0].iter)

        return ListComp(r, elt_expr, target_expr, iter_expr)

    def build_DictComp(self, ctx: SupportContext, node: astroid.DictComp):
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset)
        if (len(node.generators) != 1):
            raise NotSupportedError(r, "Only a single generator is currently supported")

        if (len(node.generators[0].ifs) != 0):
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        key_expr = self(ctx, node.key)
        value_expr = self(ctx, node.value)
        target_expr = self(ctx, node.generators[0].target)
        iter_expr = self(ctx, node.generators[0].iter)

        return DictComp(r, key_expr, value_expr, target_expr, iter_expr)

    def build_Starred(self, ctx: SupportContext, node: astroid.Starred)->Starred:
        r = ctx.make_range(node.lineno, node.col_offset, node.col_offset + 1)
        return Starred(r, self(ctx, node.value))

class Builder(StmtLibMixin, ExprLibMixin):
    def __init__(self):
        super(Builder, self).__init__()

    def __call__(self, ctx: _sources.SourceContext, node: astroid.NodeNG)->CNode:
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise UnsupportedNodeError(ctx, node)
        return method(ctx, node)

builder = Builder()
def script(obj: Callable):
    """
    Performs scripting process.

    The class loads an entire tree,


    :param obj:
    :return:
    """