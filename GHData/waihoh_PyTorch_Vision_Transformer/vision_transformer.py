import torch
import torch.nn as nn
from patch_embeddings import PatchEmbedding
from block import BuildingBlock


class VisionTransformer(nn.Module):
    def __init__(self, image_size=384,
                 patch_size=16,
                 input_channels=3,
                 num_classes=100,
                 embedding_dims=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 include_bias=True,
                 dropout_p=0.5,
                 attention_p=0.5):
        """
        :param image_size: int, the size of the image assuming that the image is square, i.e. height = width
        :param patch_size: int, size of the batch assuming that it is square
        :param input_channels: int, 1 for gray-scale and 3 for RGB channels
        :param num_classes: int, number of classes
        :param embedding_dims: int, the dimension of the embedding layer
        :param depth: int, number of BuildingBlock
        :param num_heads: int, number of heads of the attention
        :param mlp_ratio: float, to compute size (in the nearest integer) of 1st hidden layer in MLP
        :param include_bias: bool, variable to include bias or not for query, key, and value of the attention
        :param dropout_p: float, probability of dropout for the projection (Patch Embedding Layer)
        :param attention_p: float, probability of dropout for the attention
        """
        super().__init__()
        self.path_embedding = PatchEmbedding(image_size=image_size,
                                             patch_size=patch_size,
                                             input_channels=input_channels,
                                             embedding_dims=embedding_dims)
        self.cls = nn.Parameter(data=torch.zeros(1, 1, embedding_dims))
        # to get the exact position of a given patch in the image
        self.positional_embeddings = nn.Parameter(data=torch.zeros(1, 1 + self.path_embedding.num_patches, embedding_dims))
        self.pos_drop = nn.Dropout(p=dropout_p)

        self.blocks = nn.ModuleList(modules=[BuildingBlock(dim=embedding_dims,
                                                           num_heads=num_heads,
                                                           mlp_ratio=mlp_ratio,
                                                           include_bias=include_bias,
                                                           dropout_p=dropout_p,
                                                           attention_p=attention_p) for _ in range(depth)])

        self.norm = nn.LayerNorm(normalized_shape=embedding_dims, eps=1e-6)
        self.head = nn.Linear(in_features=embedding_dims, out_features=num_classes)

    def forward(self, x):
        """
        :param x: shape (n_samples, in_channels, img_size, img_size)
        """
        n_samples = x.shape[0]
        x = self.path_embedding(x)
        cls = self.cls.expand(n_samples, -1, -1)  # shape (n_samples, 1, embedding_dims)
        x = torch.cat((cls, x), dim=1)  # concatenation -> shape (n_samples, 1 + n_patches, embedding_dims)
        x = x + self.positional_embeddings
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        cls_final = x[:, 0]

        x = self.head(cls_final)

        return x


if __name__ == "__main__":
    model = VisionTransformer()
    print(model)

    test_input = torch.randn(size=(2, 3, 384, 384))
    output = model.forward(test_input)
    print(output.shape)
