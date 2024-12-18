import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional

class VisualizationManager:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')

    def plot_loss_curves(self,
                         train_losses: List[float],
                         val_losses: Optional[List[float]] = None,
                         title: str = 'Loss Curves'):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()

    def plot_latent_space(self,
                          latent_vectors: torch.Tensor,
                          labels: torch.Tensor,
                          epoch: int):
        # Reduce dimensionality for visualization if needed
        if latent_vectors.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_vectors.cpu().numpy())
        else:
            latent_2d = latent_vectors.cpu().numpy()

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                              c=labels.cpu().numpy(), cmap='tab10')
        plt.colorbar(scatter)
        plt.title(f'Latent Space Visualization (Epoch {epoch})')
        plt.savefig(self.output_dir / f'latent_space_epoch_{epoch}.png')
        plt.close()

        # Also log to tensorboard
        self.writer.add_embedding(
            latent_vectors,
            metadata=labels,
            global_step=epoch
        )

    def plot_reconstructions(self,
                             original: torch.Tensor,
                             reconstructed: torch.Tensor,
                             epoch: int,
                             num_samples: int = 5):
        # Plot original vs reconstructed samples
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))

        for i in range(num_samples):
            axes[0, i].plot(original[i].cpu().numpy())
            axes[0, i].set_title('Original')
            axes[1, i].plot(reconstructed[i].cpu().numpy())
            axes[1, i].set_title('Reconstructed')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'reconstructions_epoch_{epoch}.png')
        plt.close()

    def plot_training_metrics(self, metrics: Dict[str, float], step: int):
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'metrics/{metric_name}', value, step)

    def close(self):
        self.writer.close()