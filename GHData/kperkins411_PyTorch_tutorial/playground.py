import torch

# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)


class myclass():
    def __init__(self, kernel=3):
        self.kernel = kernel

    def getKernel(self):
        return self.kernel

    @staticmethod
    def getSKernel():
        return 7

kernel = 5
A = myclass(5)
b = A.getKernel()

c= myclass.getSKernel()
pass
