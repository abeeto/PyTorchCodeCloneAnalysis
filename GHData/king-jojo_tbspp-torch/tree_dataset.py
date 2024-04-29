import torch
import numpy as np
import random
from node_map import NODE_SIZE

BATCH_SIZE = 1

class TreeDataSet(object):
    def __init__(self, trees, labels, vector_lookup):
        self.trees = trees
        self.labels = labels
        self.vector_lookup = vector_lookup

    def data_gen(self):
        nodes, children, labels = [], [], []
        samples = 0
        for n, c, l in self.bfs_gen(self.trees, self.labels, self.vector_lookup):
            nodes.append(n)
            children.append(c)
            labels.append(l)
            samples += 1
            if samples >= BATCH_SIZE:
                yield self.padding(nodes, children, labels)
                nodes, children, labels = [], [], []
                samples = 0

        if nodes:
            yield self.padding(nodes, children, labels)

    def bfs_gen(self, trees, labels, vector_lookup):
        """ Apply breath first search to save """
        label_lookup = {label: i for i, label in enumerate(labels)}

        for tree in trees:
            node_vec, children = [], []
            label = label_lookup[tree['label']]

            queue = [(tree['tree'], -1)]
            while queue:
                node, parent_index = queue.pop(0)
                node_index = len(node_vec)
                queue.extend([(child, node_index) for child in node['children']])
                children.append([])
                if parent_index > -1:
                    children[parent_index].append(node_index)
                node_vec.append(vector_lookup[node['node']])

            yield (node_vec, children, label)

    def padding(self, node_list, children_list, label_list):
        """ padding to make same size """
        if not node_list:
            return [], [], []
        max_nodes = max([len(x) for x in node_list])
        max_children = max([len(x) for x in children_list])
        child_len = max([len(x) for y in children_list for x in y])

        nodes = [x + [NODE_SIZE] * (max_nodes - len(x)) for x in node_list]
        children = [x + ([[]] * (max_children - len(x))) for x in children_list]
        children = [[x + [0] * (child_len - len(x)) for x in sample] for sample in children_list]

        return nodes, children, label_list

    # def one_hot(self, index, length):
    #     return [1 if i == index else 0 for i in range(length)]
