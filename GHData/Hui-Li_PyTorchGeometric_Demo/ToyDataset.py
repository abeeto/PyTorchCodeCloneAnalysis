from collections import defaultdict

from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data, Batch


def get_first_nonzero_indices(data):
    """
        Find the indices of first nonzero element in each row
        https://stackoverflow.com/a/60202801
    :param data:
    :return:
    """
    idx = torch.arange(data.shape[1], 0, -1)
    tmp2 = data * idx
    indices = torch.argmax(tmp2, 1, keepdim=True)

    return indices


def get_position_emb(seq, unique_nodes):
    """
        Get position embedding for each unique node in the sequence
        Example: seq=[1, 3, 3, 2, 3], unique_nodes=[1, 2, 3], return=[4, 1, 2]
    :param seq:
    :param unique_nodes:
    :return:
    """
    reversed_seq = seq.flip(0)
    a = unique_nodes[..., None] == reversed_seq
    return get_first_nonzero_indices(a).flatten()


def test_position_emb():

    seq = torch.Tensor([1, 3, 3, 2, 2, 1])
    unique_nodes, inverse_indices = torch.unique(seq, return_inverse=True)
    reversed_seq = seq.flip(0)

    print("reversed_seq: ", reversed_seq)
    print("unique_nodes: ", unique_nodes)
    print("unique_nodes[..., None]: ", unique_nodes[..., None])
    print("unique_nodes[..., None] == seq.flip(0): ", unique_nodes[..., None] == seq.flip(0))

    a = unique_nodes[..., None] == seq.flip(0)

    print("a = unique_nodes[..., None] == seq.flip(0): ", a)
    print("get_first_nonzero_indices(a).flatten(): ", get_first_nonzero_indices(a).flatten())

    print("------test_position_emb()-----------")

    seq1 = torch.Tensor([1, 3, 0, 2])
    unique_nodes1, inverse_indices1 = torch.unique(seq1, return_inverse=True)
    print("seq1: ", seq1)
    print("unique_nodes1: ", unique_nodes1)
    print("get_position_emb(seq1, unique_nodes1): ", get_position_emb(seq1, unique_nodes1))

    seq2 = torch.Tensor([1, 3, 3, 2, 2, 1])
    unique_nodes2, inverse_indices2 = torch.unique(seq2, return_inverse=True)
    print("seq2: ", seq2)
    print("unique_nodes2: ", unique_nodes2)
    print("get_position_emb(seq2, unique_nodes2): ", get_position_emb(seq2, unique_nodes2))

    seq3 = torch.Tensor([1, 2, 3, 4])
    unique_nodes3, inverse_indices3 = torch.unique(seq3, return_inverse=True)
    print("seq3: ", seq3)
    print("unique_nodes3: ", unique_nodes3)
    print("get_position_emb(seq3, unique_nodes3): ", get_position_emb(seq3, unique_nodes3))


if __name__ == "__main__":
    test_position_emb()


def construct_weighted_graph(seq_data):
    """

    :param seq_data: list of batch_size, each has seq_len
    """
    # batch_size x seq_len
    seq_data = torch.stack(seq_data, dim=0)
    batch_size = seq_data.shape[0]
    graphs = []

    for seq_index in range(batch_size):
        # unique_node_ids: unique node ids in ascending order
        # pos2nodeid: index (of each node in the sequence) in the unique_node_ids
        # Example: input is [1, 3, 2, 3], unique_node_ids is [1, 2, 3], pos2nodeid is [0, 2, 1, 2]
        unique_node_ids, pos2nodeid = torch.unique(seq_data[seq_index], return_inverse=True)

        pos_emb = get_position_emb(seq_data[seq_index], unique_node_ids)

        # undirected edges, i.e., add twice, from start to end and from end to start
        start_nodes = torch.cat([pos2nodeid[:-1], pos2nodeid[1:]], dim=0)
        end_nodes = torch.cat([pos2nodeid[1:], pos2nodeid[:-1]], dim=0)
        edge_index = torch.stack([start_nodes, end_nodes], dim=0)

        # ToDo: slow?
        edge_weight = defaultdict(int)

        for start_node, end_node in zip(start_nodes.tolist(), end_nodes.tolist()):
            edge_weight[(start_node, end_node)] += 1

        # Must be float
        edge_attr = torch.FloatTensor(
            [edge_weight[(start_nodes[i].item(), end_nodes[i].item())] for i in range(start_nodes.shape[0])])


        # We can add more custom attributes to the constructor of Data
        # key should not contain "index" or "face". Otherwise they will be merged.
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        G = Data(x=unique_node_ids, edge_index=edge_index, edge_attr=edge_attr, num_nodes=unique_node_ids.shape[0],
                 pos_emb=pos_emb, last_node_pos=pos2nodeid[-1])

        graphs.append(G)


    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    # Construct a large graph
    # adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple
    # isolated subgraphs), and node and target features are simply concatenated in the node dimension
    graphs = Batch.from_data_list(graphs)

    return graphs


class GraphData(Dataset):

    def __init__(self, raw_data):
        self.raw_data = torch.LongTensor(raw_data)

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, index):
        return self.raw_data[index]


