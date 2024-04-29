# This code has been taken from the following source and modified, organized in class functions, for more clarity.
#
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from test.TestPytorch import TestPytorch


def main():
    test = TestPytorch()
    test.explore_dataset()
    test.train_model()
    test.test_model()
    test.show_performance()


main()
