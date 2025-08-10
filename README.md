# Toy LLM for Remote GPU Testing

A character-level transformer trained on Shakespeare for testing remote GPU setups and ML pipelines.

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

4. **Generate text:**
   ```bash
   python generate.py
   ```

## Features

- **Character-level transformer** (384d, 6 layers, ~2.5M parameters) 
- **Shakespeare dataset** - automatically downloaded, no setup needed
- **GPU detection & benchmarking** - perfect for testing remote setups  
- **Text generation** - produces coherent Shakespeare-style completions
- **Interactive generation** - test different prompts and temperatures

## Model Details

- **Architecture**: GPT-style decoder-only transformer
- **Context length**: 256 characters  
- **Vocabulary**: ~65 unique characters
- **Training time**: ~10-20 minutes on modern GPUs
- **Memory usage**: ~1-2GB GPU memory

## Configuration

Edit `config.py` to adjust:
- Model size (`n_embd`, `n_layer`, `n_head`, `block_size`)
- Training params (`batch_size`, `learning_rate`, `max_iters`)
- GPU settings (`device`, `compile`)

## Example Output

```
Prompt: "ROMEO: But soft, what light"

Generated: "ROMEO: But soft, what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief"
```

## Remote GPU Testing

Perfect for validating:
- CUDA installation and GPU visibility
- Memory allocation and training speed  
- Model checkpointing and resume capability
- End-to-end ML pipeline functionality

Fast enough for quick validation while being realistic enough to catch real issues.