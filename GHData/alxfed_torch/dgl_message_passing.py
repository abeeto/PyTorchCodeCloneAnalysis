# -*- coding: utf-8 -*-
"""...
"""
import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl


def message_func(edges):
    return {'pr' : edges.src['pr'] / edges.src['deg']}


def reduce_func(nodes):
    msgs = torch.sum(nodes.mailbox['pr'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pr' : pv}


def pagerank_naive(g):
    # Phase #1: send out messages along all edges.
    a = g.edges()
    b = g.nodes()
    for u, v in zip(*g.edges()):
        g.send((u, v))
    # Phase #2: receive messages to compute new PageRank values.
    for v in g.nodes():
        g.recv(v)


if __name__ == '__main__':
    N = 3
    DAMP = 0.85
    K = 10  # number of iterations
    g = nx.nx.erdos_renyi_graph(N, 1)
    g = dgl.DGLGraph(g)
    nx.draw(g.to_networkx(), node_size=1000, node_color=[[.5, .5, .5, ]])
    plt.show()

    g.ndata['pr'] = torch.ones(N) / N
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()

    # Register the message function and reduce function

    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)

    outcome = pagerank_naive(g)
    print('\ndone')