import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_scatter import scatter

from WGAT import WGATConv


class Seq2Graph(nn.Module):

    def __init__(self, emb_dim, hidden_size, num_heads, dropout, node_num, weighted, predict_layer_type):
        super(Seq2Graph, self).__init__()

        self.weighted = weighted
        self.predict_layer_type = predict_layer_type

        if weighted:
            self.graph_layer1 = WGATConv(in_channels=emb_dim, out_channels=hidden_size, heads=num_heads, dropout=dropout)
            self.graph_layer2 = WGATConv(in_channels=hidden_size, out_channels=hidden_size, heads=num_heads, dropout=dropout)
        else:
            self.graph_layer1 = GATConv(in_channels=emb_dim, out_channels=hidden_size, heads=num_heads, dropout=dropout,
                                        concat=True, add_self_loops=False)
            self.graph_layer2 = GATConv(in_channels=hidden_size * num_heads, out_channels=hidden_size, heads=num_heads,
                                        dropout=dropout,
                                        concat=False, add_self_loops=False)

        if predict_layer_type == 1:
            self.pred_layer = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=node_num), nn.Sigmoid())
        else:  # SR-GNN
            self.W1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.W2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.W3 = nn.Linear(in_features=hidden_size, out_features=1)
            self.W4 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)

        self.node_embs = nn.Embedding(num_embeddings=node_num, embedding_dim=emb_dim)



    def predict(self, graph_batch, hs):
        """
            use the average hidden state of all nodes in a sequence as the sequence representation
        :param graph_batch:
        :param hs:  node_num x hidden_size
        """
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        # batch_size x hidden_size
        hs = scatter(src=hs, index=graph_batch, dim=0, reduce="mean")

        # batch_size x node_num
        scores = self.pred_layer(hs)

        return scores


    def predict2(self, graph_batch, hs, last_node_pos):
        """
            draw the attention between each node and the last node (SR-GNN)
        :param graph_batch:
        :param hs:
        :param last_node_pos:
        :return:
        """
        # https://pytorch.org/docs/stable/generated/torch.bincount.html
        bins = tuple(torch.bincount(graph_batch))

        # Back to individual graphs
        # varying length
        # list of batch_size: varying_node_num x hidden_size
        batch_unique_node_hs = torch.split(hs, bins)

        # ToDo: slow?
        # batch_size x max_seq_len x hidden_size
        batch_unique_node_hs = torch.nn.utils.rnn.pad_sequence(batch_unique_node_hs, batch_first=True, padding_value=0)

        batch_indices = torch.arange(batch_unique_node_hs.shape[0], dtype=torch.long)

        # batch_size x hidden_size
        last_node_hs = batch_unique_node_hs[batch_indices, last_node_pos]

        # [batch_size, max_len, hidden_size] + [batch_size, 1, hidden_size] -> [batch_size, max_len, hidden_size] -> [batch_size, max_len, 1]
        atted_hs = self.W3(torch.tanh(self.W1(batch_unique_node_hs) + self.W2(last_node_hs.unsqueeze(1))))

        # [batch_size, max_len, hidden_size] -> [batch_size, hidden_size]
        atted_hs = torch.sum(batch_unique_node_hs * atted_hs, dim=1)

        # batch_size x hidden_size
        hs = torch.tanh(self.W4(torch.cat([atted_hs, last_node_hs], dim=1)))

        node_embs = self.node_embs.weight

        # [batch_size, hidden_size] x [emb_dim, node_num] = [batch_size, node_num]
        scores = torch.matmul(hs, node_embs.T)

        return scores


    def forward(self, graphs):

        print("graphs.x:", graphs.x)
        print("graphs.batch:", graphs.batch)
        print("graphs.edge_index:", graphs.edge_index)
        print("graphs.edge_attr:", graphs.edge_attr)
        print("graphs.num_nodes:", graphs.num_nodes)
        print("graphs.pos_emb:", graphs.pos_emb)
        print("graphs.last_node_pos:", graphs.last_node_pos)

        # node_num x emb_size
        hs = self.node_embs(graphs.x)
        # add position embedding
        # [node_num x emb_size] + [node_num x 1]
        hs = hs + graphs.pos_emb.reshape(hs.shape[0], 1)

        if self.weighted:
            # node_num x hidden_size
            hs = self.graph_layer1(hs, graphs.edge_index, graphs.edge_attr)
            # node_num x hidden_size
            hs = self.graph_layer2(hs, graphs.edge_index, graphs.edge_attr)
        else:
            # node_num x (hidden_size * num_heads)
            hs = self.graph_layer1(hs, graphs.edge_index)
            # node_num x hidden_size
            hs = self.graph_layer2(hs, graphs.edge_index)

        if self.predict_layer_type == 1:
            scores = self.predict(graphs.batch, hs)
        else:
            scores = self.predict2(graphs.batch, hs, graphs.last_node_pos)

        return scores
