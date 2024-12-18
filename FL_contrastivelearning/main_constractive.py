import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import logging
import os
from torch.utils.data import DataLoader
import numpy as np

from models.vae import VAE
from models.loss import ContrastiveLoss
from utils.logging_utils import LoggerManager
from utils.visualization import VisualizationManager
from VaE_dataloader import create_vae_dataloaders
from config import ContrastiveConfig, TrainingConfig


class PrototypeTrainer:
    def __init__(self, vae_model, config, train_config, logger, vis_manager):
        """
        Initialize trainer with VAE model and necessary configurations.
        """
        self.vae = vae_model
        self.encoder = self.vae.encoder  # Extract encoder part
        self.config = config
        self.device = train_config.device
        self.logger = logger
        self.vis_manager = vis_manager

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.criterion = ContrastiveLoss(temperature=config.temperature)
        self.global_prototypes = {}
        self.local_prototypes = {}
        self.best_accuracy = 0.0

    def compute_global_prototype(self, train_loaders):
        """
        Compute global prototype from all training data across attack types.
        """
        self.encoder.eval()
        all_features = []

        with torch.no_grad():
            for attack_type, loader in train_loaders.items():
                for data, _ in loader:
                    data = data.to(self.device)
                    features = self.encoder(data)[0]  # Get encoder features
                    all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        global_prototype = all_features.mean(dim=0)
        return global_prototype

    def compute_local_prototypes(self, train_loaders):
        """
        Compute local prototypes for each attack type.
        """
        self.encoder.eval()
        local_prototypes = {}

        with torch.no_grad():
            for attack_type, loader in train_loaders.items():
                features_list = []
                for data, _ in loader:
                    data = data.to(self.device)
                    features = self.encoder(data)[0]
                    features_list.append(features)

                attack_features = torch.cat(features_list, dim=0)
                local_prototypes[attack_type] = attack_features.mean(dim=0)

        return local_prototypes

    def train_epoch(self, train_loaders, global_prototype):
        """
        Train for one epoch using both global and local prototypes.
        """
        self.encoder.train()
        total_loss = 0
        num_batches = 0

        for attack_type, loader in train_loaders.items():
            local_prototype = self.local_prototypes[attack_type]

            for batch_idx, (data, _) in enumerate(loader):
                data = data.to(self.device)
                features = self.encoder(data)[0]

                global_loss = self.criterion(features, global_prototype.unsqueeze(0))
                local_loss = self.criterion(features, local_prototype.unsqueeze(0))

                loss = global_loss + local_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'Attack: {attack_type}, Batch [{batch_idx}/{len(loader)}], '
                        f'Loss: {loss.item():.4f}'
                    )

        return total_loss / num_batches

    def evaluate(self, test_loaders):
        """
        Evaluate model using prototype-based classification.
        """
        self.encoder.eval()
        results = {}

        with torch.no_grad():
            for attack_type, loader in test_loaders.items():
                correct = 0
                total = 0

                for data, labels in loader:
                    data = data.to(self.device)
                    features = self.encoder(data)[0]

                    # Compare with all local prototypes
                    similarities = {}
                    for proto_type, prototype in self.local_prototypes.items():
                        sim = torch.cosine_similarity(
                            features,
                            prototype.unsqueeze(0).expand(features.size(0), -1),
                            dim=1
                        )
                        similarities[proto_type] = sim

                    # Predict using highest similarity
                    predictions = []
                    for i in range(features.size(0)):
                        sims = {k: v[i].item() for k, v in similarities.items()}
                        pred = max(sims.items(), key=lambda x: x[1])[0]
                        predictions.append(pred == attack_type)

                    correct += sum(predictions)
                    total += len(predictions)

                accuracy = correct / total
                results[attack_type] = accuracy
                self.logger.info(f'Test Accuracy for {attack_type}: {accuracy:.4f}')

        return results


def main():
    # Parse arguments and setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/contrastive')
    args = parser.parse_args()

    config = ContrastiveConfig()
    train_config = TrainingConfig()

    logger_manager = LoggerManager(args.output_dir, 'contrastive_training')
    logger = logger_manager.get_logger()
    vis_manager = VisualizationManager(args.output_dir)

    # Load VAE model
    vae_model = VAE(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim
    ).to(train_config.device)

    vae_model.load_state_dict(torch.load(args.vae_model_path))

    ROOT = train_config.data_root
    attack_types = [
        attack.split(".")[0][:-10]
        for attack in os.listdir(ROOT)
        if attack.endswith('.parquet')
    ]

    # Create trainer instance
    trainer = PrototypeTrainer(
        vae_model=vae_model,
        config=config,
        train_config=train_config,
        logger=logger,
        vis_manager=vis_manager
    )

    # Create dataloaders for all attack types
    all_train_loaders = {}
    all_test_loaders = {}

    for attack in attack_types:
        dataloaders = create_vae_dataloaders(
            data_root=train_config.data_root,
            attack_type=attack,
            batch_size=config.batch_size,
        )
        all_train_loaders[attack] = dataloaders['train']
        all_test_loaders[attack] = dataloaders['test']

    # Compute global and local prototypes
    logger.info("Computing global prototype...")
    global_prototype = trainer.compute_global_prototype(all_train_loaders)

    logger.info("Computing local prototypes...")
    trainer.local_prototypes = trainer.compute_local_prototypes(all_train_loaders)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        train_loss = trainer.train_epoch(all_train_loaders, global_prototype)
        test_results = trainer.evaluate(all_test_loaders)

        # Visualization
        vis_manager.plot_training_metrics({
            'train_loss': train_loss,
            **{f'accuracy_{k}': v for k, v in test_results.items()}
        }, epoch)

        # Save best model
        avg_accuracy = sum(test_results.values()) / len(test_results)
        if avg_accuracy > trainer.best_accuracy:
            trainer.best_accuracy = avg_accuracy
            torch.save(
                trainer.encoder.state_dict(),
                Path(args.output_dir) / 'best_prototype_encoder.pt'
            )

    # Final evaluation
    logger.info("\nFinal Evaluation Results:")
    final_results = trainer.evaluate(all_test_loaders)
    for attack_type, accuracy in final_results.items():
        logger.info(f"{attack_type}: {accuracy:.4f}")

    vis_manager.close()


if __name__ == "__main__":
    main()