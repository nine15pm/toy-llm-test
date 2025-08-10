# Toy LLM Training Setup

A minimal PyTorch transformer for testing remote GPU setups and ML pipelines.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run setup check:**
   ```bash
   python setup.py
   ```

3. **Train the model:**
   ```bash
   python train.py
   ```

## Features

- **Tiny transformer model** (128d, 4 layers) - trains quickly
- **GPU detection & benchmarking** - perfect for testing remote setups  
- **Synthetic dataset** - no external data dependencies
- **Training metrics** - loss tracking and model checkpointing
- **Generation testing** - verify model works end-to-end

## Configuration

Edit `config.py` to adjust:
- Model size (`n_embd`, `n_layer`, `n_head`)
- Training params (`batch_size`, `learning_rate`, `max_iters`)
- GPU settings (`device`, `compile`)

## Remote GPU Testing

This setup is ideal for testing:
- CUDA installation and GPU visibility
- Memory allocation and training speed
- Model checkpointing and resume
- Basic ML pipeline validation

The model trains in ~2-5 minutes on modern GPUs, making it perfect for quick validation of remote environments.