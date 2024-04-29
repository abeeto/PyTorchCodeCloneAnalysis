import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, input_channels=3, embedding_dims=768):
        """
        :param image_size: int, the size of the image assuming that the image is square, i.e. height = width
        :param patch_size: int, size of the batch assuming that it is square
        :param input_channels: int, 1 for gray-scale and 3 for RGB channels
        :param embedding_dims: int, the dimension of the embedding layer
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedding_dims = embedding_dims

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(in_channels=input_channels, out_channels=embedding_dims,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: shape (n_samples, input_channels, image_size, image_size), i.e. square image.
        """
        projection = self.projection(x)
        projection = projection.flatten(start_dim=2)  # shape (n_samples, embedding_dim, sqr(n_patches), sqr(n_patches))
        projection = projection.transpose(1, 2)  # shape (n_samples, n_patches, embedding_dim)
        return projection


if __name__ == "__main__":
    test_projection = PatchEmbedding(image_size=10, patch_size=2, input_channels=3)
    # note num_patches = (image_size // patch_size)**2, thus for image_size=10, patch_size=2, num_patches=25

    import torch
    test_input = torch.randn(size=(2, 3, 10, 10))
    output = test_projection.forward(test_input)
    print(output.shape)
