import torch
import tempfile
import os
import subprocess

# Credit to: Peng Wu
# TSAllType := TSType | TSModuleType
# 
# TSType := TSMultiType | TSPrimitiveType | TSStructuralType | 
#           TSBuiltinNominalType | TSCustomNominalType
# TSModuleType := "torch.nn.Module" and subclasses
# 
# TSMultiType := "Any"
# TSPrimitiveType := "int" | "float" | "double" | "bool" | "str" | NoneType
# TSStructualType :=  TSTuple | TSList | TSDict | TSOptional | TSFuture |
#                     TSRRef
# TSBuiltinNominalType := TSTensor | "torch.collections.namedtuple" | 
#                   "torch.device" | "torch.stream" | "torch.dtype" | 
#                   "torch.nn.ModuleList" | "torch.nn.ModuleDict" | ...
# TSCustomNominalType := TSEnum | TSClass | TSInterface
# 
# TSTuple := "Tuple" "[" (TSType ",")+ "]"
# TSList := "List" "[" TSType "]"
# TSOptional := "Optional" "[" TSType "]"
# TSFuture := "Future" "[" TSType "]"
# TSRRef := "RRef" "[" TSType "]"
# TSDict := "Dict" "[" KeyType "," TSType "]"
# KeyType := "str" | "int" | "float" | "bool" | TSTensor| "Any"
# 
# TSTensor := "torch.tensor" and subclasses
# TSInterface := classes w/ attr "____torch_script_interface__"

DummyDeclarations = """
class DummyNNModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

import enum
class DummyEnum(enum.Enum):
    TORCH = "torch"
    SCRIPT = "script"

@torch.jit.script
class DummyClass():
    def __init__(self, v: int) -> None:
        self.v = v

    def exported(self) -> int:
        return self.v

@torch.jit.interface
class DummyInterface():
    def __init__(self, v: int) -> None:
        pass

    def forward(self) -> int:
        pass

import collections
DummyNamedTuple = collections.namedtuple("DummyNamedTuple", ["value"])
"""

class TSTypes:

    @staticmethod
    def TSAllType(nest_depth):
        res = []
        res.extend(TSTypes.TSType(nest_depth))
        res.extend(TSTypes.TSModuleType(nest_depth))
        return res

    @staticmethod
    def TSModuleType(nest_depth):
        return ["DummyNNModule"]

    @staticmethod
    def TSType(nest_depth):
        res = []
        res.extend(TSTypes.TSMultiType())
        res.extend(TSTypes.TSPrimitiveType())
        res.extend(TSTypes.TSStructuralType(nest_depth))
        res.extend(TSTypes.TSBuiltinNominalType())
        res.extend(TSTypes.TSCustomNominalType())
        return res

    @staticmethod
    def TSMultiType():
        return ["Any"]

    @staticmethod
    def TSBuiltinNominalType():
        res = []
        res.extend(TSTypes.TSTensor())
        res.append("DummyNamedTuple")
        res.append("torch.device")
        res.append("torch.Stream")
        res.append("torch.dtype")
        # TODO: ModuleDict and ModuleList can not be parsed by annotation
        #res.append("torch.nn.ModuleList")
        #res.append("torch.nn.ModuleDict")
        return res

    @staticmethod
    def TSTensor():
        return ["torch.Tensor"]

    @staticmethod
    def TSCustomNominalType():
        res = []
        res.extend(TSTypes.TSEnum())
        res.extend(TSTypes.TSClass())
        res.extend(TSTypes.TSInterface())
        return res

    @staticmethod
    def TSEnum():
        return ["DummyEnum"]

    @staticmethod
    def TSClass():
        return ["DummyClass"]

    @staticmethod
    def TSInterface():
        return ["DummyInterface"]

    @staticmethod
    def TSPrimitiveType():
        # TODO: How to express None type?
        # TODO: double is not a valid annotation
        return ["int", "float", "bool", "str"]

    @staticmethod
    def TSStructuralType(nest_depth):
        res = []
        res.extend(TSTypes.TSTuple(nest_depth))
        res.extend(TSTypes.TSList(nest_depth))
        res.extend(TSTypes.TSDict(nest_depth))
        res.extend(TSTypes.TSOptional(nest_depth))
        res.extend(TSTypes.TSFuture(nest_depth))
        res.extend(TSTypes.TSRRef(nest_depth))
        return res

    def TSTuple(nest_depth):
        if nest_depth == 0:
            return []
        else:
            nest_depth = nest_depth - 1

        # TODO: Here we are testing Tuple of only 1 element, which is same as
        # multiple-element tuples in theory.
        res = []
        for t in TSTypes.TSType(nest_depth):
            res.append("Tuple[" + t + "]")
        return res

    def TSList(nest_depth):
        if nest_depth == 0:
            return []
        else:
            nest_depth = nest_depth - 1

        res = []
        for t in TSTypes.TSType(nest_depth):
            res.append("List[" + t + "]")
        return res


    def TSOptional(nest_depth):
        if nest_depth == 0:
            return []
        else:
            nest_depth = nest_depth - 1

        res = []
        for t in TSTypes.TSType(nest_depth):
            res.append("Optional[" + t + "]")
        return res

    def TSFuture(nest_depth):
        # TODO: Future is not supported yet
        return []

    def TSRRef(nest_depth):
        # TODO: RRef is not supported yet
        return []

    def TSDict(nest_depth):
        if nest_depth == 0:
            return []
        else:
            nest_depth = nest_depth - 1

        res = []
        for k in TSTypes.TSKeyType():
            for v in TSTypes.TSType(nest_depth):
                res.append("Dict[" + k + ", " + v + "]")
        return res

    def TSKeyType():
        res = ["str", "int", "float", "bool", "Any"]
        res.extend(TSTypes.TSTensor())
        return res

def ConstructTestFileBoilerPlate():
    source = """
import torch
from typing import Any, Tuple, List, Optional, Dict
    """

    source += DummyDeclarations 
    return source

def ConstructPassThroughCase(type_str: str):
    source = """
def fn(x: {type_str}) -> {type_str}:
    return x

try:
    torch.jit.script(fn)
    print("{type_str} case successfully compiled")
except Exception as e:
    print("{type_str} case failed to compile")
    print("Error is: " + str(e))
    """.format(type_str=type_str)

    return source

def main():
    name = ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        name = f.name
        f.write(ConstructTestFileBoilerPlate())
        for t in TSTypes.TSAllType(nest_depth=3):
            f.write(ConstructPassThroughCase(t))
    subprocess.call(["python3", f.name])

if __name__ == "__main__":
    main()
	
