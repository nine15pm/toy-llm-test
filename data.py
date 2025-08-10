import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # Generate synthetic data: simple arithmetic sequences
        # Pattern: a, b, a+b, a*2, b*2, (a+b)*2, ...
        np.random.seed(42 if split == 'train' else 1337)
        
        # Generate base numbers
        n_samples = 10000 if split == 'train' else 1000
        self.data = []
        
        for _ in range(n_samples):
            a, b = np.random.randint(1, 50, 2)
            sequence = [
                a % config.vocab_size,
                b % config.vocab_size, 
                (a + b) % config.vocab_size,
                (a * 2) % config.vocab_size,
                (b * 2) % config.vocab_size,
                ((a + b) * 2) % config.vocab_size
            ]
            
            # Extend to block_size with some pattern variation
            while len(sequence) < config.block_size:
                last_val = sequence[-1]
                next_val = (last_val + np.random.randint(1, 10)) % config.vocab_size
                sequence.append(next_val)
            
            self.data.append(sequence[:config.block_size])
        
        self.data = torch.tensor(self.data, dtype=torch.long)
        print(f"Loaded {split} dataset: {len(self.data)} samples, vocab_size={config.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[:-1]  # Input sequence
        y = sample[1:]   # Target sequence (shifted by 1)
        return x, y

def get_dataloaders(config):
    train_dataset = ToyDataset(config, 'train')
    val_dataset = ToyDataset(config, 'val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    return train_loader, val_loader