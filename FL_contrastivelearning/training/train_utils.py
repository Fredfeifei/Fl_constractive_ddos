import torch
from typing import Dict, Tuple
from torch.utils.data import DataLoader

class TrainingManager:
    def __init__(self, model, optimizer, criterion, device, logger, vis_manager):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.vis_manager = vis_manager

        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self,
                    train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data[0].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.criterion(recon_batch, data, mu, log_var)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}'
                )

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self,
                 val_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0
        all_reconstructions = []
        all_originals = []
        all_latents = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data = data[0].to(self.device)

                recon_batch, mu, log_var = self.model(data)
                loss = self.criterion(recon_batch, data, mu, log_var)

                total_loss += loss.item()

                all_reconstructions.append(recon_batch)
                all_originals.append(data)
                all_latents.append(mu)
                all_labels.append(labels)

        vis_data = {
            'reconstructions': torch.cat(all_reconstructions),
            'originals': torch.cat(all_originals),
            'latents': torch.cat(all_latents),
            'labels': torch.cat(all_labels)
        }

        avg_loss = total_loss / len(val_loader)
        return avg_loss, vis_data