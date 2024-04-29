

import time
import numpy as np
from scipy import spatial
from sklearn.covariance import graphical_lasso
from pyunlocbox import functions, solvers
import gower
import torch

from scipy import sparse

# -----------------------------------------------------------------------------
""" Functions used in :func:`learn_graph.log_degree_barrier method` """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight2degmap(N, array=False):
    r"""
    Generate linear operator K such that W @ 1 = K @ vec(W).
    Parameters
    ----------
    N : int
        Number of nodes on the graph
    Returns
    -------
    K : function
        Operator such that K(w) is the vector of node degrees
    Kt : function
        Adjoint operator mapping from degree space to edge weight space
    array : boolean, optional
        Indicates if the maps are returned as array (True) or callable (False).
    Examples
    --------
    >>> import learn_graph
    >>> K, Kt = learn_graph.weight2degmap(10)
    Notes
    -----
    Used in :func:`learn_graph.log_degree_barrier method`.
    """
    

    Ne = int(N * (N - 1) / 2)  # Number of edges
    row_idx1 = np.zeros((Ne, ))
    row_idx2 = np.zeros((Ne, ))
    count = 0
    for i in np.arange(1, N):
        row_idx1[count: (count + (N - i))] = i - 1
        row_idx2[count: (count + (N - i))] = np.arange(i, N)
        count = count + N - i
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)

def weight2degmap_fast(N, array=False):
    Ne = int(N * (N - 1) / 2)  # Number of edges
    
    row_idx2 = np.tile(np.arange(0,N), (N,1) )
    row_idx1 = (np.copy(row_idx2)).transpose()
    row_idx1 = row_idx1[np.triu_indices(len(row_idx1),k=1)]
    row_idx2 = row_idx2[np.triu_indices(len(row_idx2),k=1)]
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)

def validation_reg(X, valid_adj,dist_type='sqeuclidean', alpha = 1,beta=1, step=0.5,
                       w0=None, maxit=1000, rtol=1e-5, retall=False,
                       verbosity='NONE'):
    r"""
    Learn graph by imposing a log barrier on the degrees
    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1}  + \beta \| W - W_o \|_{F}^{2}`,
    where :math:`Z` is a pairwise distance matrix, and :math:`\mathcal{W}_m`
    is the set of valid symmetric weighted adjacency matrices.
    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    valid_adj : array-like
        An N*N-1/2 data matrix representing a un-refined version of the
        adjacency matrix to be learnt
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    beta : float, optional
        Regularization parameter controlling the density of the graph
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.
    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.
    """
    valid_adj = valid_adj[np.triu_indices(len(valid_adj),k=1)]
    # Parse X
    N = X.shape[0]
    if (dist_type == 'gower'):
        z = np.nan_to_num(gower.gower_matrix(X), copy=True, nan=0.0, posinf= np.inf)
        np.fill_diagonal(z, 0)
        z = spatial.distance.squareform(z)
    else:
        z = torch.nn.functional.pdist((torch.from_numpy(X)).to(device), p=2)
        z = z.cpu().detach().numpy()

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = valid_adj #np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    K, Kt =  weight2degmap_fast(N)
    norm_K = np.sqrt(2 * (N - 1))

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * ((torch.dot((torch.from_numpy(w)).to(device), (torch.from_numpy(z)).to(device))).cpu().detach().numpy())
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: - alpha * np.sum(np.log(np.maximum(
        np.finfo(np.float64).eps, K(w))))
    f2._prox = lambda d, gamma: np.maximum(
        0, 0.5 * (d + np.sqrt(d**2 + (4 * alpha * gamma))))


    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum((w- valid_adj) **2)
    f3._grad = lambda w: 2 * beta * (w - valid_adj)
    lipg = 2 * beta

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W

def make_sparse(W, edges, rr):
    W_plt = W
    for i in range(0, len(W)):
        W_plt[i][i] = 0
    a_1d = W.flatten()
    a = np.zeros(W.shape)
# Find the indices in the 1D array
    idx_1d = (a_1d.argsort())[-int(edges*(1-rr)):]

# convert the idx_1d back into indices arrays for each dimension
    x_idx, y_idx = np.unravel_index(idx_1d, W.shape)
    for x, y, in zip(x_idx, y_idx):
        a[x][y] = 1
    return W * a