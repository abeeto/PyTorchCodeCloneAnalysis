# document https://docs.python.org/3.9/library/ast.html#compiler-flags
import ast
from ast_utility import iter_child_nodes, iter_code
src='''
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
'''


ast_node = ast.parse(src, "msg.log", mode="exec")
c_code=iter_code(ast_node)
print(c_code)
print(ast.dump(ast_node))