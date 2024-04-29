import astroid
import inspect
from typing import Callable, List




#########


class BasePreprocessor()
    """
    This class is capable of preprocessing
    an input function or class to be 
    more torchscript compatible.
        
    Preprocessing consists of unraveling complex
    programs into many simpler components in a 
    virtual model. The preprocessor ultimately 
    just walks the tree of the function it is
    given, making many decisions as to whether or
    not to break the given code block down further. 
    
    The base preprocessor simply contains the logic for this,
    in the form of transforms. Transforms are a function which
    accept an astroid node, and return either the node again
    or a new node, and may also place trees into the virtual
    module list.
    """
    def __init__(self, transforms: Optional[List[Callable]] = None):
        self.transforms = transforms
    def preprocess(self, object: Callable):
        code = inspect.getsource(object)


def preprocess(item: Callable):
    """

    This function is capable


    :param item:
    :return:
    """


    virtual_module: List[astroid.NodeNG] = []




class preprocessor()
    """
    
    The preprocessor is capable of 
    performing a variety of useful 
    
    
    """



    def __init__(self):
        self.virtual_module =

    def inline_nonmethod_functions(self,):