import torch
import torch.optim as optim
from pathlib import Path
import argparse
import logging
import os

from models.vae import VAE
from models.loss import VAELoss
from utils.logging_utils import LoggerManager
from utils.visualization import VisualizationManager
from VaE_dataloader import create_vae_dataloaders
from config import VAEConfig, TrainingConfig

def train_vae_single_attack(
        attack_type: str,
        model: VAE,
        dataloaders: dict,
        config: VAEConfig,
        train_config: TrainingConfig,
        logger: logging.Logger,
        vis_manager: VisualizationManager,
        output_dir: Path
) -> VAE:

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = VAELoss(beta=config.beta)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloaders['train']):
            data = data.to(train_config.device)

            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = criterion(recon_batch, data, mu, log_var)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 4500 == 0:
                logger.info(
                    f'Attack: {attack_type}, Epoch: {epoch}, Batch: {batch_idx}, '
                    f'Loss: {loss.item():.4f}'
                )

        avg_train_loss = train_loss / (len(dataloaders['train'])*config.batch_size)
        logger.info(f'Attack: {attack_type}, Epoch: {epoch}, Average Train Loss: {avg_train_loss:.4f}')

        if epoch % 3 ==0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in dataloaders['val']:
                    data = data.to(train_config.device)
                    recon_batch, mu, log_var = model(data)
                    loss = criterion(recon_batch, data, mu, log_var)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (len(dataloaders['val'])*config.batch_size)

            logger.info(
                f'Attack: {attack_type}, Epoch: {epoch}, '
                f'Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}'
            )

            vis_manager.plot_training_metrics({
                f'{attack_type}/train_loss': avg_train_loss,
                f'{attack_type}/val_loss': avg_val_loss
            }, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    output_dir / 'vae_model.pt'
                )
                logger.info(f"Saved new best model with validation loss: {avg_val_loss:.2f}")
            else:
                patience_counter += 1
                logger.info(f"Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return model

def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VAEConfig()
    train_config = TrainingConfig()

    logger_manager = LoggerManager(output_dir, 'vae_training')
    logger = logger_manager.get_logger()
    vis_manager = VisualizationManager(output_dir)

    model_path = output_dir / 'vae_model.pt'

    ROOT = TrainingConfig.data_root
    attack_types = [attack.split(".")[0][:-10] for attack in os.listdir(ROOT)]

    for attack_type in attack_types:
        if attack_type != "syn": continue;

        logger.info(f"\nStarting training for attack type: {attack_type}")
        dataloaders = create_vae_dataloaders(
            data_root=args.data_root,
            attack_type=attack_type,
            batch_size=config.batch_size,
            test_drop=TrainingConfig.test_drop
        )

        model = VAE(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim
        ).to(train_config.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")

        if model_path.exists():
            logger.info(f"Loading previous model to continue training")
            model.load_state_dict(torch.load(model_path))

        model = train_vae_single_attack(
            attack_type=attack_type,
            model=model,
            dataloaders=dataloaders,
            config=config,
            train_config=train_config,
            logger=logger,
            vis_manager=vis_manager,
            output_dir=output_dir
        )

        if TrainingConfig.test_drop:
            continue;
        else:
            model.eval()
            final_test_loss = 0
            criterion = VAELoss(beta=config.beta)

            with torch.no_grad():
                for data, _ in dataloaders['test']:
                    data = data.to(train_config.device)
                    recon_batch, mu, log_var = model(data)
                    loss = criterion(recon_batch, data, mu, log_var)
                    final_test_loss += loss.item()

            avg_test_loss = final_test_loss / len(dataloaders['test'])
            logger.info(f"Final test loss for {attack_type}: {avg_test_loss:.4f}")

    logger.info("Training completed for all attack types!")
    vis_manager.close()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                       default='D:/FL_contrastivelearning/data/processed',
                       help='Directory containing parquet files')
    parser.add_argument('--output_dir', type=str,
                       default='D:/FL_contrastivelearning/outputs/vae',
                       help='Directory to save outputs')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)