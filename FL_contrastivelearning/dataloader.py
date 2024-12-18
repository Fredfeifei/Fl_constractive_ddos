from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import math

class ContrastiveTrafficDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            attack_type: str,
            mode: str = 'train',
            seed: int = 42,
            num_augmentations: int = 2
    ):
        """
        Args:
            data_root: Root directory containing parquet files
            attack_type: Name of the attack parquet file
            mode: 'train' or 'test'
            seed: Random seed for reproducibility
            num_augmentations: Number of augmented views per sample
        """

        super().__init__()
        self.attack_type = attack_type
        self.mode = mode
        self.num_augmentations = num_augmentations
        np.random.seed(seed)

        # Load data
        self.data, self.labels = self._load_attack_data(data_root, attack_type)
        print(f"Loaded {len(self.data)} total samples for {attack_type}")

        self.indices = self._create_bootstrapped_indices()

        self.train_indices, self.test_indices = self._create_split()

        self.active_indices = self.train_indices if mode == 'train' else self.test_indices
        print(f"{mode.capitalize()} set size: {len(self.active_indices)}")

        self._print_distribution()

    def _load_attack_data(self, data_root: str, attack_type: str) -> Tuple[np.ndarray, np.ndarray]:

        file_path = Path(data_root) / f"{attack_type}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for attack type: {attack_type}")

        df = pd.read_parquet(file_path)
        label_column = 'label' if 'label' in df.columns else 'attack_type'

        if 'similarhttp' in df.columns:
            df = df.drop('similarhttp', axis=1)

        feature_columns = [col for col in df.columns if col != label_column]

        features = df[feature_columns].values
        labels = (df[label_column] != 0).astype(int)

        return features, labels.values

    def _create_bootstrapped_indices(self) -> np.ndarray:

        """Create balanced dataset by bootstrapping benign class"""

        attack_indices = np.where(self.labels == 1)[0]
        benign_indices = np.where(self.labels == 0)[0]

        print(f"\nInitial class distribution:")
        print(f"Benign samples: {len(benign_indices)}")
        print(f"Attack samples: {len(attack_indices)}")

        # Calculate how many times we need to replicate benign samples
        n_replications = math.ceil(len(attack_indices) / len(benign_indices))

        # Bootstrap benign indices
        bootstrapped_benign = []
        for _ in range(n_replications):
            bootstrapped_benign.extend(benign_indices)

        # Take exactly the number we need to match attack samples
        bootstrapped_benign = np.array(bootstrapped_benign[:len(attack_indices)])

        # Combine indices
        balanced_indices = np.concatenate([bootstrapped_benign, attack_indices])
        np.random.shuffle(balanced_indices)

        print(f"\nAfter bootstrapping:")
        print(f"Total samples: {len(balanced_indices)}")
        print(f"Benign samples (bootstrapped): {len(bootstrapped_benign)}")
        print(f"Attack samples: {len(attack_indices)}")

        return balanced_indices

    def _create_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create stratified train/test split"""
        train_ratio = 0.7

        balanced_labels = self.labels[self.indices]
        train_indices = []
        test_indices = []

        for label in [0, 1]:
            label_indices = self.indices[balanced_labels == label]
            np.random.shuffle(label_indices)

            split_idx = int(train_ratio * len(label_indices))
            train_indices.extend(label_indices[:split_idx])
            test_indices.extend(label_indices[split_idx:])

        return np.array(train_indices), np.array(test_indices)

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Simple feature augmentation with scaling and noise
        """
        augmented = features.copy()

        scale = np.random.uniform(0.9, 1.1)
        augmented = augmented * scale

        noise = np.random.normal(0, 0.01, features.shape)
        augmented = augmented + noise

        return augmented

    def _print_distribution(self):
        """Print class distribution"""
        labels = self.labels[self.active_indices]
        n_benign = np.sum(labels == 0)
        n_attack = np.sum(labels == 1)
        total = len(labels)

        print(f"\nClass distribution in {self.mode} set:")
        print(f"Benign traffic: {n_benign} samples ({n_benign / total * 100:.2f}%)")
        print(f"Attack traffic: {n_attack} samples ({n_attack / total * 100:.2f}%)")

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        data_idx = self.active_indices[idx]
        original_features = self.data[data_idx]

        if self.mode == 'train':
            # Generate augmented views for contrastive learning
            augmented_views = [
                self._augment_features(original_features)
                for _ in range(self.num_augmentations)
            ]
            augmented_views = [torch.FloatTensor(view) for view in augmented_views]
        else:
            # For test set, return original features without augmentation
            augmented_views = [torch.FloatTensor(original_features)]

        label = torch.LongTensor([self.labels[data_idx]])
        return augmented_views, label


def create_contrastive_dataloaders(
        data_root: str,
        attack_types: List[str],
        batch_size: int = 32,
        num_augmentations: int = 2
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Create DataLoaders for contrastive learning with bootstrapped sampling
    """
    dataloaders = {mode: {} for mode in ['train', 'test']}

    for mode in ['train', 'test']:
        for attack_type in attack_types:
            dataset = ContrastiveTrafficDataset(
                data_root=data_root,
                attack_type=attack_type,
                mode=mode,
                num_augmentations=num_augmentations
            )

            dataloaders[mode][attack_type] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(mode == 'train'),
                num_workers=2,
                pin_memory=True
            )

    return dataloaders