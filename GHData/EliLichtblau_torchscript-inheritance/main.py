#import torch
from typing import Union, List, Type, Any, Tuple, Optional, NamedTuple, Callable, Dict

from class_to_string import inline_class



class superstr:
    pass
class supertuple:
    pass
class superint:
    pass

class One:
    def __init__(self, random_input: superstr, r2: supertuple, r3: superint, kwargs: Optional[Any]=None):
        self.oneVar = 1
        self.common = 10
        r = r2*r3
        x = random_input**2
        some_var = kwargs
    def one_method(self):
        print("One method")
    
    def overloaded(self):
        pass



#import torch


#@Inherit
class Two(One):
    def __init__(self, TWO_INIT1: str, TWO_INIT2: Tuple, TWO_INIT3: int, TWO_INIT_KWARG: Optional[Any] =None):
        # my comment
        super().__init__(TWO_INIT1, TWO_INIT2, TWO_INIT3, kwargs = "test")
        self.twoVar = 2

    def two_method(self):
        print("Some bullshit")

    def overloaded(self):
        print("now i do stuff")
#GLOBAL_DICT = dict() # key = tuple(type signautes), (tuple(int, int, double) -> compiled class)




if __name__ == "__main__":
    #tmp = inline_super(Two.__init__)
    #twoInit = parse_def(Two.__init__)
    #t = Two(5)
    #print(t.twoVar)
    src = inline_class(Two)
    breakpoint()