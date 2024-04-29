import os

if __name__ == "__main__":
    # for files (e.g. data)
    def pathing():
        entires = os.listdir(".")
        for entry in entires:
            print(entry)
        print(os.listdir("D:/Hochschule RT/Masterthesis/Coding/PyTorchTut"))
        # test prints:
        # path = "." is equal to "root" <- root is where the .py is located at
        print(os.listdir("."))

    # pathing()

# epoch: all data from the dataset

# this is a tuple!
strings = ("hey", "Domi")
print(strings[0:2])
lists = ["hey", "Domi"]
print(lists[0:2])

import torch
t = torch.ones(3,2)
print(t)
sum = t.sum()
print(sum)
print(sum.item())