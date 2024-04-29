import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, dim, num_heads, include_bias, attention_dropout=0.5, projection_dropout=0.5):
        """
        :param dim: int, Input/Output dimensions
        :param num_heads: int, number of heads of the attention
        :param include_bias: bool, variable to include bias or not for query, key, and value of the attention
        :param attention_dropout: float, probability of dropout for the attention
        :param projection_dropout: float, probability of dropout for the projection (Patch Embedding Layer)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.include_bias = include_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # linear mapping take in token embedding and generate query, key and a value (reason for dim * 3)
        self.linear_layer = nn.Linear(in_features=dim, out_features=dim*3, bias=include_bias)
        self.projection = nn.Linear(in_features=dim, out_features=dim)

        self.attention_drop = nn.Dropout(p=self.attention_dropout)
        self.projection_drop = nn.Dropout(p=self.projection_dropout)

    def forward(self, x):
        """
        :param x: tensor, shape (n_samples, n_patches + 1, dim), num_patches + 1 for the 0 class token (from the paper)
        :return: output
        """

        # extract the dimension
        n_samples, n_tokens, dim = x.shape
        linear = self.linear_layer(x)  # shape: (n_samples, n_patches + 1, dim * 3)
        linear = linear.reshape(n_samples, n_tokens, 3, self.num_heads, self.head_dim)
        linear = linear.permute(2, 0, 3, 1, 4)  # shape (3, n_samples, num_heads, n_patches + 1, head_dim)
        # to extract query, key, value
        query = linear[0]
        key = linear[1]
        value = linear[2]

        key_transpose = key.transpose(-2, -1)  # shape (num_samples, num_heads, head_dim, n_patches + 1)
        # NOTE: a @ b invokes a.__matmul__(b)
        query_key = (query @ key_transpose) * self.scale  # From Attention is all you need (transformers)
        # NOTE: to generate a discreet probability distribution that sums up to one for weighted average
        attention = query_key.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, )
        attention = self.attention_drop(attention)
        weighted_average = attention @ value
        weighted_average_transpose = weighted_average.transpose(1, 2)
        # NOTE: to flat the last 2 dims for concatenation, shape (n_samples, n_patches + 1, head_dim)
        weighted_average_flat = weighted_average_transpose.flatten(2)

        output = self.projection(weighted_average_flat)  # shape (n_samples, n_patches + 1, dim)
        output = self.projection_drop(output)

        return output


if __name__ == "__main__":
    # Note: self.head_dim = dim // num_heads. so ensure dim is divisible by num_heads
    test_attention = AttentionModel(dim=24, num_heads=6, include_bias=True)
    print(test_attention)

    import torch
    test_input = torch.randn(size=(3, 2, 24))  # the last dimension must match the dim parameter in AttentionModel
    output = test_attention.forward(test_input)
    print(output.shape)
