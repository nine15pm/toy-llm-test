import torch

class Config:
    # Model architecture
    vocab_size = 65  # Will be updated by dataset (Shakespeare has ~65 chars)
    n_embd = 384     # Embedding dimension - scaled up for better text generation
    n_layer = 6      # Number of transformer layers - increased
    n_head = 6       # Number of attention heads - increased
    block_size = 256 # Context length - increased for better text understanding
    dropout = 0.1
    
    # Training
    batch_size = 32  # Increased for better training stability
    learning_rate = 3e-4
    max_iters = 5000  # More iterations for text learning
    eval_interval = 100
    eval_iters = 20   # More eval iterations for better estimates
    
    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile = False  # Set to True for torch.compile optimization
    
    # Logging
    wandb_log = False
    wandb_project = 'toy-llm'
    
    def __str__(self):
        return f"Config(device={self.device}, vocab_size={self.vocab_size}, n_embd={self.n_embd}, n_layer={self.n_layer})"