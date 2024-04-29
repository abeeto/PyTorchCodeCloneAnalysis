import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def wrap(y, dtype='float'):
    y_wrap = Variable(torch.from_numpy(y))
    if dtype=='float':
        y_wrap = y_wrap.float()
    elif dtype == 'long':
        y_wrap = y_wrap.long()
if torch.cuda.is_available():
    y_wrap = y_wrap.cuda()
    return y_wrap


def unwrap(y_wrap):
    if y_wrap.is_cuda:
        y = y_wrap.cpu().data.numpy()
    else:
        y = y_wrap.data.numpy()
    return y


def wrap_X(X):
    X_wrap = copy.deepcopy(X)
    for jet in X_wrap:
        jet["content"] = wrap(jet["content"])
    return X_wrap


def unwrap_X(X_wrap):
    X_new = []
    for jet in X_wrap:
        jet["content"] = unwrap(jet["content"])
        X_new.append(jet)
    return X_new


##Batchization of the jets using LOUPPE'S code
def batch(jets):
    jet_children = []  # [n_nodes, 2]=> jet_children[nodeid, 0], jet_children[nodeid, 1]
    offset = 0
    for j, jet in enumerate(jets):
        tree = np.copy(jet["tree"])
        tree[tree != -1] += offset
        jet_children.append(tree)
        offset += len(tree)
    
    jet_children = np.vstack(jet_children)
    jet_contents = torch.cat([jet["content"] for jet in jets], 0) # [n_nodes, n_features]
    n_nodes = offset

    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32)
    level_children[:, [0, 2]] -= 1
    
    inners = []   # Inner nodes at level i
    outers = []   # Outer nodes at level i
    offset = 0
    
    for jet in jets:
        queue = [(jet["root_id"] + offset, -1, True, 0)]
        
        while len(queue) > 0:
            node, parent, is_left, depth = queue.pop(0)
            
            if len(inners) < depth + 1:
                inners.append([])
            if len(outers) < depth + 1:
                outers.append([])
            
            if jet_children[node, 0] != -1: # Inner node
                inners[depth].append(node)
                position = len(inners[depth]) - 1
                is_leaf = False
                
                queue.append((jet_children[node, 0], node, True, depth + 1))
                queue.append((jet_children[node, 1], node, False, depth + 1))
            
            else:   # Outer node
                outers[depth].append(node)
                position = len(outers[depth]) - 1
                is_leaf = True
            
            if parent >= 0: # Register node at its parent
                if is_left:
                    level_children[parent, 0] = position
                    level_children[parent, 1] = is_leaf
                else:
                    level_children[parent, 2] = position
                    level_children[parent, 3] = is_leaf

        offset += len(jet["tree"])
    
    # Reorganize levels[i] so that inner nodes appear first, then outer nodes
    levels = []
    n_inners = []
    contents = []
    
    prev_inner = np.array([], dtype=int)
    
    for inner, outer in zip(inners, outers):
        n_inners.append(len(inner))
        inner = np.array(inner, dtype=int)
        outer = np.array(outer, dtype=int)
        level = np.concatenate((inner, outer))
        level = torch.from_numpy(level)
        if torch.cuda.is_available(): level = level.cuda()
        levels.append(level)
        
        left = prev_inner[level_children[prev_inner, 1] == 1]
        level_children[left, 0] += len(inner)
        right = prev_inner[level_children[prev_inner, 3] == 1]
        level_children[right, 2] += len(inner)
        
        contents.append(jet_contents[levels[-1]])
        
        prev_inner = inner
    
    # levels: list of arrays
    #     levels[i][j] is a node id at a level i in one of the trees
    #     inner nodes are positioned within levels[i][:n_inners[i]], while
    #     leaves are positioned within levels[i][n_inners[i]:]
    #
    # level_children: array of shape [n_nodes, 2]
    #     level_children[node_id, 0] is the position j in the next level of
    #         the left child of node_id
    #     level_children[node_id, 1] is the position j in the next level of
    #         the right child of node_id
    #
    # n_inners: list of shape len(levels)
    #     n_inners[i] is the number of inner nodes at level i, accross all
    #     trees
    #
    # contents: array of shape [n_nodes, n_features]
    #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
    #     or node layers[i][j]
    
    level_children = torch.from_numpy(level_children).long()
    n_inners = torch.from_numpy(np.array(n_inners)).long()
    if torch.cuda.is_available():
        level_children = level_children.cuda()
        n_inners = n_inners.cuda()

    return (levels, level_children[:, [0, 2]], n_inners, contents)



