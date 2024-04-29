import Layer 
import Fn as fn 

class Module():
    def __init__(self):
        self.layers = []
        self.output = None

    def forward(self, inputs):
        pass 

    def __call__(self, inputs):
        output = self.forward(inputs)
        self.output = output
        return output
