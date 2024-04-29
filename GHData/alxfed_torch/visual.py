# -*- coding: utf-8 -*-
"""...
"""
import networkx as nx
import dgl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    g_nx = nx.petersen_graph()
    g_dgl = dgl.DGLGraph(g_nx)

    plt.subplot(121)
    nx.draw(g_nx, with_labels=True)
    plt.subplot(122)
    nx.draw(g_dgl.to_networkx(), with_labels=True)

    plt.show()
    print('\ndone')