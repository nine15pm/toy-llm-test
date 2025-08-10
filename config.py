import torch

class Config:
    # Model architecture
    vocab_size = 1000  # Small vocab for toy model
    n_embd = 128       # Embedding dimension
    n_layer = 4        # Number of transformer layers
    n_head = 4         # Number of attention heads
    block_size = 64    # Context length
    dropout = 0.1
    
    # Training
    batch_size = 16
    learning_rate = 3e-4
    max_iters = 1000
    eval_interval = 100
    eval_iters = 10
    
    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile = False  # Set to True for torch.compile optimization
    
    # Logging
    wandb_log = False
    wandb_project = 'toy-llm'
    
    def __str__(self):
        return f"Config(device={self.device}, vocab_size={self.vocab_size}, n_embd={self.n_embd}, n_layer={self.n_layer})"