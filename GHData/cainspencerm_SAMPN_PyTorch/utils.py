import math
from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import dgl
from rdkit import Chem
from pprint import pprint

# Metric utils.

def pearson_cor(targets: List[float], preds: List[float]) -> float:
    """
    Computes the Pearson correlation coefficient.
    
    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed Pearson correlation coefficient.
    """

    targets, preds = np.array(targets).flatten(), np.array(preds).flatten()
    return np.corrcoef(targets, preds)[0, 1]

def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))

_metrics_QSAR = {
    'rmse': rmse,
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'r2': r2_score,
    'pearson': pearson_cor,
}

_metrics_QSARPlus = {
    'rmse': rmse,
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
}

def get_metrics(model_type):
    """
    Gets the metrics for a model.

    :param model_type: The type of model.
    :return: A list of metrics.
    """

    if model_type == 'QSAR':
        return _metrics_QSAR
    elif model_type == 'QSARPlus':
        return _metrics_QSARPlus
    else:
        raise ValueError('Invalid model type.')


# Torch utils.

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim

    target = source.index_select(dim=0, index=index.view(-1))
    target = target.view(final_size)

    return target

def initialize_weights(model: torch.nn.Module):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """

    for name, param in model.named_parameters():
        if param.dim() == 1:
            torch.nn.init.constant_(param, 0)
        else:
            torch.nn.init.xavier_normal_(param)


# Graph utils.

def get_graph(mol: Chem.Mol) -> dgl.DGLGraph:
    """
    Gets the graph of a molecule.

    :param mol: A molecule string.
    :return: A list of lists of integers representing the graph.
    """

    # Get the edge list (starting_nodes, ending_nodes).
    starting_nodes, ending_nodes = [], []
    for bond in mol.GetBonds():

        # Start to end.
        starting_nodes.append(bond.GetBeginAtomIdx())
        ending_nodes.append(bond.GetEndAtomIdx())

        # End to start.
        starting_nodes.append(bond.GetEndAtomIdx())
        ending_nodes.append(bond.GetBeginAtomIdx())

    starting_nodes, ending_nodes = np.array(starting_nodes), np.array(ending_nodes)
    
    # Generate the graph.
    graph = dgl.graph((starting_nodes, ending_nodes))

    return graph