#!/usr/bin/env python3

import torch
from config import Config
from model import ToyLLM
from data import TextDataset

def load_model(checkpoint_path='best_model.pt'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    model = ToyLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Training iteration: {checkpoint['iter_num']}")
    print(f"Validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, config

def main():
    try:
        model, config = load_model()
    except FileNotFoundError:
        print("No saved model found. Please train the model first with: python train.py")
        return
    
    # Create dataset for tokenization
    dataset = TextDataset(config, 'train')
    
    print("\n=== Text Generation Demo ===")
    print("Model will complete your text. Type 'quit' to exit.")
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            continue
        
        try:
            # Encode prompt
            context = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=config.device)
            
            # Truncate if too long
            if context.size(1) > config.block_size:
                context = context[:, -config.block_size:]
                print("(Prompt truncated to fit context window)")
            
            print(f"\nGenerating completion...")
            
            # Generate with different temperatures
            temperatures = [0.3, 0.8, 1.2]
            
            for temp in temperatures:
                generated = model.generate(
                    context, 
                    max_new_tokens=150, 
                    temperature=temp, 
                    top_k=40
                )
                
                generated_text = dataset.decode(generated[0].tolist())
                prompt_end = len(prompt)
                completion = generated_text[prompt_end:]
                
                print(f"\n--- Temperature {temp} ---")
                print(f"{prompt}{completion}")
                
        except Exception as e:
            print(f"Generation error: {e}")

if __name__ == "__main__":
    main()