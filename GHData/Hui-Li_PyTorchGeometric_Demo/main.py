from torch.utils.data import DataLoader
from Seq2Graph import Seq2Graph
from ToyDataset import GraphData, construct_weighted_graph

if __name__ == "__main__":

    raw_data = [
                [1,3,0,2],
                [2,3,1,3],
                [2,3,2,1],
                [0,3,2,2]
               ]

    batch_size = 2
    emb_dim = 8
    hidden_size = 8
    num_heads = 2
    dropout = 0
    node_num = 4

    graph_data = GraphData(raw_data=raw_data)

    # collate_fn is the function used for batching
    dataset_loader = DataLoader(dataset=graph_data, batch_size=batch_size,
                                collate_fn=construct_weighted_graph, shuffle=False)

    model1 = Seq2Graph(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, node_num=node_num, weighted=False, predict_layer_type=1)

    print("--------- model 1 using GAT and averaging node hidden states as the sequence representation ----------")
    for graph_data in dataset_loader:
        scores = model1(graphs=graph_data)
        print(scores)


    model2 = Seq2Graph(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, node_num=node_num, weighted=True, predict_layer_type=1)

    print("--------- model 2 using Weighted GAT  and averaging node hidden states as the sequence representation ----------")
    for graph_data in dataset_loader:
        scores = model2(graphs=graph_data)
        print(scores)


    model3 = Seq2Graph(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, node_num=node_num, weighted=False, predict_layer_type=2)

    print("--------- model 3 using GAT and the method of SR-GNN as the sequence representation ----------")
    for graph_data in dataset_loader:
        scores = model3(graphs=graph_data)
        print(scores)


    model4 = Seq2Graph(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, node_num=node_num, weighted=True, predict_layer_type=1)

    print("--------- model 4 using Weighted GAT and the method of SR-GNN as the sequence representation ----------")
    for graph_data in dataset_loader:
        scores = model4(graphs=graph_data)
        print(scores)
