import sys

__author__ = "H.D. 'Chip' McCullough IV"


def parseinput(strng1, strng2):
    if "a" in strng1.lower():
        a = False
    elif "a" in strng2.lower():
        a = True
    else:
        print("'A' was not found in either string.")
        sys.exit(2)
    if "b" in strng1.lower():
        b = False
    elif "b" in strng2.lower():
        b = True
    else:
        print("'B' was not found in either string.")
        sys.exit(3)
    if "c" in strng1.lower():
        c = False
    elif "c" in strng2.lower():
        c = True
    else:
        print("'C' was not found in either string.")
        sys.exit(4)
    if "d" in strng1.lower():
        d = False
    elif "d" in strng2.lower():
        d = True
    else:
        print("'D' was not found in either string.")
        sys.exit(5)
    if "p" in strng1.lower():
        p = False
    elif "p" in strng2.lower():
        p = True
    else:
        print("'P' was not found in either string.")
        sys.exit(6)

    return (a, b, c, d, p)
