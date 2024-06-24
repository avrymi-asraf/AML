import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import time
from tqdm import tqdm
import plotly.express as px
from utilitis import *

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=200):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2),  # (batch_size, 512, 1, 1)
        )

        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2),  # (batch_size, 128, 2, 2)
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1
            ),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (batch_size, 1, 28, 28)
        )

    def reparameterize(self, mu, logvar):
        device = next(self.parameters()).device
        var = torch.exp(logvar * 0.5)
        return torch.randn_like(mu).to(device) * var + mu

    def encode(self, x):
        """
        calculte the ?
        """
        x = self.encoder(x)
        # add average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 1, 1)
        z = self.decoder(z)
        return z

    def forward(self, x):
        """ """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(input, target, mu, logvar):
    return F.mse_loss(input, target) + torch.mean(
        mu.pow(2) + torch.exp(logvar) + 0.5 * logvar - 1, dim=1
    )


def train_model_VAE(
    model, data_loader, epochs, examples, device="cpu", live_result=True
):
    optimazer = optim.Adam(model.parameters())
    record_data = pd.DataFrame({"epoch_loss": float()}, index=range(epochs))
    examples = examples.to(device)
    model.train()

    reconstruct_images = {"source": model(examples)[0].detach().cpu()}
    for epoch in range(epochs):
        start_time = time.time()
        loss_epoch = 0.0
        for x, _ in tqdm(data_loader):

            x = x.to(device)
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)

            loss.backward()
            optimazer.step()
            optimazer.zero_grad()

            loss_epoch += loss.item()
            record_data.iloc[epoch] = [loss_epoch]

        if live_result:
            clear_output(wait=True)
            px.line(record_data).show()
        if (epoch + 1) % 5 == 0:
            reconstruct_images[str(epoch + 1)] = model(examples)[0].detach().cpu()
            px.imshow(reconstruct_images[str(epoch + 1)].squeeze(1), facet_col=0)
        print(f"epoch {epoch+1}, loss: {loss_epoch:.5f}")
    return record_data, reconstruct_images


def train_model_OL():
    torch.rand(19).detach()


def main():
    train_loader = import_MNIST_dataset(test=False)
    examples = import_MNIST_examples(train_loader)
    model = ConvVAE()
    rd, im = train_model_VAE(model,train_loader, 10, examples,live_result=False)
