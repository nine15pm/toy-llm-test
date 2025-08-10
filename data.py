import torch
import numpy as np
import os
import requests
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # Download Shakespeare dataset if not exists
        data_file = 'shakespeare.txt'
        if not os.path.exists(data_file):
            print("Downloading Shakespeare dataset...")
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            response = requests.get(url)
            with open(data_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Downloaded {len(response.text)} characters")
        
        # Read and process text
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character-level tokenizer
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        config.vocab_size = self.vocab_size  # Update config
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        data = [self.stoi[ch] for ch in text]
        
        # Split data
        n = len(data)
        split_idx = int(0.9 * n)
        
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]
        
        print(f"Loaded {split} dataset: {len(self.data)} characters, vocab_size={self.vocab_size}")

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.config.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])
    
    def encode(self, text):
        return [self.stoi[ch] for ch in text]

def get_dataloaders(config):
    train_dataset = TextDataset(config, 'train')
    val_dataset = TextDataset(config, 'val')
    
    # Update config with actual vocab size
    config.vocab_size = train_dataset.vocab_size
    
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
    
    return train_loader, val_loader, train_dataset