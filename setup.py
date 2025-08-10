#!/usr/bin/env python3

import subprocess
import sys
import torch
import time
import psutil

def check_python_version():
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8+ recommended")
    else:
        print("✓ Python version OK")

def check_gpu_setup():
    print("\n=== GPU Setup Check ===")
    
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ PyTorch version: {torch.__version__}")
        
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPUs available: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            torch.cuda.synchronize()
            print("✓ GPU memory allocation test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU memory allocation failed: {e}")
            
    else:
        print("✗ CUDA not available - will use CPU")
        print(f"PyTorch version: {torch.__version__}")

def check_system_resources():
    print("\n=== System Resources ===")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024**3
    print(f"RAM: {memory_gb:.1f} GB ({memory.percent}% used)")
    
    # Disk space
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / 1024**3
    print(f"Disk free: {disk_free_gb:.1f} GB")

def benchmark_training_speed():
    print("\n=== Training Speed Benchmark ===")
    
    from config import Config
    from model import ToyLLM
    
    config = Config()
    config.batch_size = 8  # Smaller for benchmark
    
    print(f"Device: {config.device}")
    
    model = ToyLLM(config)
    model.to(config.device)
    
    # Generate dummy data
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size-1), device=config.device)
    y = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size-1), device=config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Warmup
    for _ in range(5):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if config.device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 20
    start_time = time.time()
    
    for _ in range(n_iters):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if config.device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    iters_per_sec = n_iters / elapsed
    
    print(f"Training speed: {iters_per_sec:.2f} iterations/second")
    print(f"Time per iteration: {elapsed/n_iters*1000:.1f} ms")

def install_dependencies():
    print("\n=== Installing Dependencies ===")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")

def main():
    print("=== Toy LLM Remote GPU Setup Check ===")
    
    check_python_version()
    
    # Try to install dependencies first
    try:
        import torch
    except ImportError:
        print("PyTorch not found, installing dependencies...")
        install_dependencies()
        import torch
    
    check_gpu_setup()
    check_system_resources()
    
    try:
        benchmark_training_speed()
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    print("\n=== Summary ===")
    print("Setup check completed. Ready for training!")
    print("Run: python train.py")

if __name__ == "__main__":
    main()