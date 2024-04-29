import sys
from ParseInput import parseinput
from TestHarness import test_harness

__author__ = "H.D. 'Chip' McCullough IV"


def main():
    agent = "astarsearch"
    str1 = sys.argv[1]
    str2 = sys.argv[2]
    state = parseinput(str1, str2)

    test_harness(state, agent)


if __name__ == '__main__':
    main()
