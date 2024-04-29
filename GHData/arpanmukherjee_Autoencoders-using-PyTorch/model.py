from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, ip_dim, h_dim, op_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ip_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, op_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(op_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, ip_dim),
            nn.ReLU(True)
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ConvolutionAE(nn.Module):
    def __init__(self):
        super(ConvolutionAE, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, X):
        encoded = self.relu(self.conv1(X))
        encoded, ind1 = self.pool(encoded)
        encoded = self.relu(self.conv2(encoded))
        encoded, ind2 = self.pool(encoded)

        decoded = self.unpool(encoded, ind2)
        decoded = self.relu(self.deconv1(decoded))
        decoded = self.unpool(decoded, ind1)
        decoded = self.relu(self.deconv2(decoded))
        return encoded, decoded
