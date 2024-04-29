import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_p=0.5):
        """
        :param in_features: int, input size
        :param hidden_features: int, size of 1st hidden layer
        :param out_features: int, output size
        :param dropout_p: float, dropout probability
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # neural network
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        :param x: tensor, shape (n_samples, n_patches + 1, in_features)
        """
        linear1 = self.layer1(x)
        gelu = self.gelu(linear1)
        gelu = self.drop(gelu)
        linear2 = self.linear2(gelu)
        output = self.drop(linear2)
        return output


if __name__ == "__main__":
    test_mlp = MLP(in_features=4, hidden_features=8, out_features=2)
    print(test_mlp)

    import torch
    test_input = torch.randn(size=(2, 4))
    output = test_mlp.forward(test_input)
    print(output.shape)
