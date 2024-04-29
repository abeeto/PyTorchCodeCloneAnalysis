# -*- coding: utf-8 -*-
""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import os
from collections import namedtuple
from models import ops

Genotype = namedtuple('Genotype', 'dag ops')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_7x1_1x7',
    'conv_1x1',
    'none',
]

abbr = {
    'none': 'NIL',
    'avg_pool_3x3': 'AVG',
    'max_pool_3x3': 'MAX',
    'skip_connect': 'IDT',
    'sep_conv_3x3': 'SC3',
    'sep_conv_5x5': 'SC5',
    'sep_conv_7x7': 'SC7',
    'dil_conv_3x3': 'DC3',
    'dil_conv_5x5': 'DC5',
    'conv_7x1_1x7': 'FC7',
    'conv_1x1':     'C11',
}

deabbr = {
    'NIL': 'none',
    'AVG': 'avg_pool_3x3',
    'MAX': 'max_pool_3x3',
    'IDT': 'skip_connect',
    'SC3': 'sep_conv_3x3',
    'SC5': 'sep_conv_5x5',
    'SC7': 'sep_conv_7x7',
    'DC3': 'dil_conv_3x3',
    'DC5': 'dil_conv_5x5',
    'FC7': 'conv_7x1_1x7',
    'C11': 'conv_1x1',
}

def set_primitives(prim):
    global PRIMITIVES
    PRIMITIVES = prim
    print('candidate ops: {}'.format(get_primitives()))

def get_primitives():
    return PRIMITIVES

def pretty_print(gene):
    pass

def to_file(gene, path):
    g_str = str(gene)
    with open(path, 'w', encoding='UTF-8') as f:
        f.write(g_str)

def from_file(path):
    if not os.path.exists(path):
        print("genotype file not found: {}".format(path))
        return Genotype(dag=None, ops=None)
    with open(path, 'r', encoding='UTF-8') as f:
        g_str = f.read()
    return from_str(g_str)

def from_str(s):
    """ generate genotype from string """
    genotype = eval(s)
    return genotype
