#!/usr/bin/env python3
import sys
import click
import ast

def run(filepath):
    with open(filepath, "r") as f:
        t = ast.parse(f.read(), filepath)
    print(t)

if __name__ == "__main__":
    run(sys.argv[1])


