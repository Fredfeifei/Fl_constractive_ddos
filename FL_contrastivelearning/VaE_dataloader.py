import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class VAETrafficDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            attack_type: str,
            mode: str = 'train',
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            seed: int = 55,
            scaler_type: str = 'standard',
            scaler: Optional[object] = None
    ):
        super().__init__()
        self.attack_type = attack_type
        self.mode = mode
        self.scaler_type = scaler_type
        self.data_root = Path(data_root)
        np.random.seed(seed)

        self.splits_dir = self.data_root / 'splits'
        self.splits_dir.mkdir(exist_ok=True)

        self.features, self.labels = self._load_data(data_root, attack_type)
        self.train_indices, self.val_indices, self.test_indices = \
            self._load_or_create_splits(attack_type, train_ratio, val_ratio)

        self.active_indices = {
            'train': self.train_indices,
            'val': self.val_indices,
            'test': self.test_indices
        }[mode]

        self.features = self._normalize_features(scaler)

        if scaler is not None:
            self.scaler = scaler

        print(f"Created {mode} dataset with {len(self.active_indices)} samples")

    def _save_indices_to_file(self, indices: np.ndarray, attack_type: str, mode: str):
        """Save indices to a text file."""
        file_path = self.splits_dir / f"{attack_type}_{mode}.txt"
        np.savetxt(file_path, indices, fmt='%d')

    def _load_indices_from_file(self, attack_type: str, mode: str) -> Optional[np.ndarray]:
        """Load indices from a text file if it exists."""
        file_path = self.splits_dir / f"{attack_type}_{mode}.txt"
        if file_path.exists():
            return np.loadtxt(file_path, dtype=int)
        return None

    def _load_or_create_splits(self, attack_type: str, train_ratio: float, val_ratio: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Load existing splits or create new ones if they don't exist."""

        train_indices = self._load_indices_from_file(attack_type, 'train')
        val_indices = self._load_indices_from_file(attack_type, 'val')
        test_indices = self._load_indices_from_file(attack_type, 'test')

        if any(split is None for split in [train_indices, val_indices, test_indices]):
            print(f"Creating new splits for {attack_type}")
            train_indices, val_indices, test_indices = self._create_splits(train_ratio, val_ratio)

            # Save the new splits
            self._save_indices_to_file(train_indices, attack_type, 'train')
            self._save_indices_to_file(val_indices, attack_type, 'val')
            self._save_indices_to_file(test_indices, attack_type, 'test')
        else:
            print(f"Loading existing splits for {attack_type}")

        return train_indices, val_indices, test_indices

    def _normalize_features(self, provided_scaler: Optional[object] = None) -> np.ndarray:
        if provided_scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")

            self.scaler.fit(self.features[self.train_indices])
        else:
            self.scaler = provided_scaler

        if self.scaler is None:
            raise ValueError("Scaler not initialized properly")

        normalized_features = self.scaler.transform(self.features)

        if self.scaler_type == 'standard':
            train_mean = np.mean(normalized_features[self.train_indices], axis=0)
            train_std = np.std(normalized_features[self.train_indices], axis=0)
            print(f"Training set - Mean: {train_mean.mean():.3f}, Std: {train_std.mean():.3f}")
        else:
            train_min = np.min(normalized_features[self.train_indices], axis=0)
            train_max = np.max(normalized_features[self.train_indices], axis=0)
            print(f"Training set - Min: {train_min.mean():.3f}, Max: {train_max.mean():.3f}")

        return normalized_features

    def get_scaler(self) -> object:
        return self.scaler

    def _load_data(self, data_root: str, attack_type: str) -> Tuple[np.ndarray, np.ndarray]:
        file_path = Path(data_root) / f"{attack_type}_processed.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for attack type: {attack_type}")

        df = pd.read_parquet(file_path)
        label_column = 'label' if 'label' in df.columns else 'attack_type'
        if 'similarhttp' in df.columns:
            df = df.drop('similarhttp', axis=1)

        feature_columns = [col for col in df.columns if col != label_column]
        features = df[feature_columns].values
        labels = (df[label_column] != 0).astype(int)

        return features, labels

    def _create_splits(self, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(len(self.features))
        train_indices = []
        val_indices = []
        test_indices = []

        for label in [0, 1]:
            label_indices = indices[self.labels == label]
            np.random.shuffle(label_indices)

            n_train = int(len(label_indices) * train_ratio)
            n_val = int(len(label_indices) * val_ratio)

            train_indices.extend(label_indices[:n_train])
            val_indices.extend(label_indices[n_train:n_train + n_val])
            test_indices.extend(label_indices[n_train + n_val:])

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.active_indices[idx]
        features = self.features[idx]
        label = self.labels[idx]

        return torch.FloatTensor(features), torch.LongTensor([label])

def create_vae_dataloaders(
        data_root: str,
        attack_type: str,
        batch_size: int = 128,
        train_ratio: float = 0.4,
        val_ratio: float = 0.1,
        seed: int = 42,
        scaler_type: str = 'standard',
        test_drop: bool = True
) -> Dict[str, DataLoader]:

    dataloaders = {}
    train_dataset = VAETrafficDataset(
        data_root=data_root,
        attack_type=attack_type,
        mode='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        scaler_type=scaler_type
    )

    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    for mode in ['val', 'test']:
        if mode == 'test' and test_drop:
            continue

        dataset = VAETrafficDataset(
            data_root=data_root,
            attack_type=attack_type,
            mode=mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            scaler_type=scaler_type
        )

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    return dataloaders