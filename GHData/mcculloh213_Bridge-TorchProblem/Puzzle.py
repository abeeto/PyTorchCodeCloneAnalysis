import sys
from TestHarness import test_harness

__author__ = "H.D. 'Chip' McCullough IV"


def main():
    agent = "8puzzle"
    f_in = sys.argv[1]
    init = []
    read = open(f_in, 'r')
    for line in read:
        init.append(line.split())
    read.close()
    test_harness(init, agent)

if __name__ == "__main__":
    main()