import numpy as np
import Fn as fn
from Var import Var

class Linear():
    def __init__(self, input_len, output_len):
        w = Var()
        w.data = np.random.randn(output_len, input_len)
        b = Var()
        b.data = np.zeros((output_len, 1))
        self.parameters = [w, b]

    def forward(self, x):
        x = fn.reshape(x, (-1, 1))
        w, b = self.parameters[0], self.parameters[1]
        z = fn.add(fn.mul(w, x), b)
        return z

    def __call__(self, x):
        return self.forward(x)

