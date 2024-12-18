import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout_rate=0.3):
        super().__init__()

        # Initial projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Processing layer (simplified)
        self.processing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Latent projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.input_projection(x)

        for residual in self.residual_layers:
            h = residual(h)

        h = self.processing_layer(h)

        return self.fc_mu(h), self.fc_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.3):
        super().__init__()

        # Initial projection from latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Residual layers
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        # Processing layer (simplified)
        self.processing_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):

        h = self.latent_projection(z)
        for residual in self.residual_layers:
            h = residual(h)

        h = self.processing_layer(h)
        return self.output_projection(h)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout_rate=0.15):
        super().__init__()

        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout_rate)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, num_layers, dropout_rate)
        self.apply(self._init_weights)

        self.noise_scale = 0.01

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, log_var):

        if self.training:
            log_var = torch.clamp(log_var, -20, 2)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        return mu

    def forward(self, x):

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return self.reparameterize(mu, log_var)

    def decode(self, z):
        return self.decoder(z)