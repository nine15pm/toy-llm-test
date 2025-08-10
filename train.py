import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
from tqdm import tqdm
import wandb

from config import Config
from model import ToyLLM
from data import get_dataloaders

def estimate_loss(model, data_loader, config):
    model.eval()
    losses = torch.zeros(config.eval_iters)
    
    for k, (x, y) in enumerate(data_loader):
        if k >= config.eval_iters:
            break
        x, y = x.to(config.device), y.to(config.device)
        
        with torch.no_grad():
            logits, loss = model(x, y)
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()

def log_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU available - using CPU")

def main():
    config = Config()
    print(f"Training configuration: {config}")
    log_gpu_info()
    
    if config.wandb_log:
        wandb.init(project=config.wandb_project)
        wandb.config.update(vars(config))
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Create model
    print("Initializing model...")
    model = ToyLLM(config)
    model.to(config.device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    iter_num = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.max_iters // len(train_loader) + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for x, y in pbar:
            if iter_num >= config.max_iters:
                break
                
            x, y = x.to(config.device), y.to(config.device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if iter_num % config.eval_interval == 0:
                train_loss = loss.item()
                val_loss = estimate_loss(model, val_loader, config)
                
                print(f"\nIter {iter_num}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if config.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }, 'best_model.pt')
            
            iter_num += 1
        
        if iter_num >= config.max_iters:
            break
    
    print("Training completed!")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    context = torch.randint(0, config.vocab_size, (1, 10), device=config.device)
    generated = model.generate(context, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"Generated sequence: {generated[0].tolist()}")
    
    if config.wandb_log:
        wandb.finish()

if __name__ == "__main__":
    main()