import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw3_utils import BASE_URL, download, GANDataset


class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=8*5*5, out_features=1)
        )

        self.apply(self._weight_init)

    def _weight_init(self, m):
        # TODO: implement weight initialization here
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # TODO: complete forward function
        x = self.conv_layers(x)
        x = x.view(-1, 8*5*5)  # flatten
        x = self.linear_layers(x)
        return x.squeeze()


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=zdim, out_features=1568),
            nn.LeakyReLU(0.2)
        )
        self.conv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.apply(self._weight_init)

    def _weight_init(self, m):
        # TODO: implement weight initialization here
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        x = self.linear_layer(z)
        x = x.view(-1, 32, 7, 7)
        x = self.conv_layers(x)
        return x


class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        real_label = torch.ones((batch_size, 1), device=self._dev)
        fake_label = torch.zeros((batch_size, 1), device=self._dev)

        # Train the discriminator with real data
        d_real = self.disc(batch_data)
        loss_real = nn.BCEWithLogitsLoss()(d_real, real_label)

        # Train the discriminator with fake data
        fake_data = self.gen(z)
        d_fake = self.disc(fake_data.detach())
        loss_fake = nn.BCEWithLogitsLoss()(d_fake, fake_label)

        loss_d = (loss_real + loss_fake) / 2

        return loss_d

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.
        Compute -\sum_z\log{D(G(z))} instead of \sum_z\log{1-D(G(z))}
        
        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        fake_data = self.gen(z)
        d_fake = self.disc(fake_data)
        loss_g = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake)))

        return loss_g

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
