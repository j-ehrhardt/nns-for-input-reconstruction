import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, latent_dims)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        return self.linear4(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 784)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.dropout(self.relu(self.linear1(z)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        x = torch.sigmoid(self.linear4(x))
        return x.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, upper, middle, lower):
        x = torch.cat([upper, middle, lower], dim=2)
        s_upper, s_middle, s_lower = upper.size(2), middle.size(2), lower.size(2)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        upper_hat = x_hat[:, :, :s_upper, :]
        middle_hat = x_hat[:, :, s_upper:s_upper+s_middle, :]
        lower_hat = x_hat[:, :, s_upper+s_middle:, :]
        return upper_hat, middle_hat, lower_hat