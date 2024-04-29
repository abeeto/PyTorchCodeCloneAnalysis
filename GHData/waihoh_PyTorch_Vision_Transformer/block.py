import torch.nn as nn
from attention import AttentionModel
from mlp import MLP


class BuildingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, include_bias=True, dropout_p=0.5, attention_p=0.5):
        """
        :param dim: int, input size
        :param num_heads: int, number of heads of the attention
        :param mlp_ratio: float, to compute size (in the nearest integer) of 1st hidden layer in MLP
        :param include_bias: bool, variable to include bias or not for query, key, and value of the attention
        :param dropout_p: float, probability of dropout for the attention
        :param attention_p: float, probability of dropout for the projection (Patch Embedding Layer)
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=1e-6)
        self.attention = AttentionModel(dim=dim, num_heads=num_heads, include_bias=include_bias,
                                        attention_dropout=attention_p, projection_dropout=dropout_p)
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=1e-6)
        self.hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=self.hidden_features, out_features=dim)

    def forward(self, x):
        """
        :param x: shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.attention(self.norm1(x))  # x = x + residual connection
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    test_building_block = BuildingBlock(dim=10, num_heads=5, mlp_ratio=2)
    print(test_building_block)

    import torch
    test_input = torch.randn(2, 3, 10)
    output = test_building_block.forward(test_input)
    print(output.shape)
