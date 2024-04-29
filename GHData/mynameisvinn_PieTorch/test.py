import numpy as np
import unittest
from nn import Tensor, Add, Multiply, Module, Relu, Pow, Loss, Optimizer, Matmul

class Test_PieTorch(unittest.TestCase):

    def setUp(self):
        self.X = Tensor(val=-2, name="X")
        self.Y = Tensor(val=5, name="Y")
        self.Q = Add(self.X, self.Y)
        self.Z = Tensor(val=-4, name="Z")
        self.F = Multiply(self.Q, self.Z)

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__() 
                self.X = Tensor(val=-2, name="X")
                self.Y = Tensor(val=5, name="Y")
                self.Z = Tensor(val=-4, name="Z")
                
            def forward(self, x):
                self.Q = Add(self.X, self.Y)
                self.F = Multiply(self.Q, self.Z)
                return self.F

        self.model = Net()
        

    def test_forward_pass(self):
        """
        to be revised since F.val is not dynamically computed at runtime
        """
        self.assertEqual(self.F.val, -12.0)

    def test_accumulated_grad(self):
        self.F.backward()  # call backward on root
        self.assertEqual(self.X.accumulated_grad, -4.0)
        self.assertEqual(self.Y.accumulated_grad, -4.0)
        self.assertEqual(self.Z.accumulated_grad, 3.0)

    def test_module(self):
        self.assertEqual(self.model(3).val, -12.0)

    def test_loss(self):
        output = self.model(0)
        self.assertEqual(output.val, -12.0)  # first feedforward

        criterion = Loss()
        loss = criterion(output, self.model.X)
        self.assertEqual(loss.val, -10.0)  # check loss

        loss.backward()
        optimizer = Optimizer(self.model.parameters(), learning_rate=1)
        optimizer.step()
        output = self.model(0)  # second feedforward
        self.assertEqual(output.val, -77.0)

    def test_matmul(self):
        x = np.array([1, 3, -5])
        y = np.array([4, -2, -1])
        z = Matmul(x, y)
        self.assertEqual(z.val, 3.0)        

if __name__ == "__main__":
    unittest.main()