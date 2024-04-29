# -*- coding: utf-8 -*-
"""...
"""
import networkx as nx
import matplotlib.pyplot as plt
import torch as tch
import dgl


def message_func(edges):
    return {'pr' : edges.src['pr'] / edges.src['deg']}


def reduce_func(nodes):
    msgs = tch.sum(nodes.mailbox['pr'], dim=1)
    pv = 1 * msgs
    return {'pr' : pv}


def pagerank_naive(g):
    # Phase #1: send out messages along all edges.
    a = G.edges()
    b = G.nodes()
    for u, v in zip(*G.edges()):
        G.send((u, v))
    # Phase #2: receive messages to compute new PageRank values.
    for v in G.nodes():
        G.recv(v)


if __name__ == '__main__':
    G = dgl.DGLGraph()
    G.add_nodes(3, {'propositions': tch.ones(3, 5)})
    G.add_nodes(1)
    print(G.ndata['propositions'])
    G.add_edges([0, 1], [1, 2], {'predicates': tch.ones(2, 5)})
    print(G.edata['predicates'])

    nx.draw(G.to_networkx(), node_size=1000, node_color=[[.5, .5, .5, ]])
    plt.show()

    N = 5
    G.ndata['pr'] = tch.ones(N) / N   # creates a property 'pr' on every node and assigns a value to it
    G.ndata['deg'] = G.out_degrees(G.nodes()).float() # creates a property 'deg' on every node and assigns

    # Register tche message function and reduce function

    G.register_message_func(message_func)
    G.register_reduce_func(reduce_func)

    outcome = pagerank_naive(G)
    print('\ndone')