import torch
torch.classes.load_library("build/libdcgan.so")
# check whether the library is loaded correctly
print(torch.classes.loaded_libraries)
# load the graph_data path and graph information
graph_data_path = "/home/datalab/data/test1"
# the number of node in the graph
num_node = 100

manager = torch.classes.my_classes.ManagerWrap(1, num_node, graph_data_path)
manager.create_static_view(manager)
DAMP = 0.85
num_iter = 10

def compute_pagerank(s):
    pg_value = torch.ones(num_node,1)
    for k in range(num_iter):
        out_message = s.scatter_gather(pg_value,"sum", s)
        pg_value = (1 - DAMP) / num_node + DAMP * out_message
    return pg_value

pg_value = compute_pagerank(manager)
print (pg_value)
num_iter = 3
pg_value = compute_pagerank(manager)
print (pg_value)



