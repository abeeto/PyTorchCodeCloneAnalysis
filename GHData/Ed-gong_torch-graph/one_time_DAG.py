import torch
torch.classes.load_library("build/libdcgan.so")
# check whether the library is loaded correctly
print(torch.classes.loaded_libraries)
# load the graph_data path and graph information
graph_data_path = "/home/datalab/data/test1"
# the number of node in the graph
num_node = 100

s = torch.classes.my_classes.ManagerWrap(1, num_node, graph_data_path)
# initial the page_rank value as all ones
pg_value = torch.ones(num_node,1)
# use the initial PR_value as message to scatter_gather once
b = s.scatter_gather(pg_value,"sum")
print (b)
# check the size of returned pg_value
print(b.shape)
