# DeepSeek-V3 Usage Examples and Tutorials

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Basic Usage Examples](#basic-usage-examples)
3. [Advanced Usage Scenarios](#advanced-usage-scenarios)
4. [Distributed Computing](#distributed-computing)
5. [Model Conversion Workflows](#model-conversion-workflows)
6. [Performance Optimization](#performance-optimization)
7. [Integration Examples](#integration-examples)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Quick Start Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/inference

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, triton, transformers, safetensors; print('All dependencies installed successfully')"
```

### Download Model Weights

```bash
# Download from Hugging Face (example)
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3 /path/to/DeepSeek-V3
```

### Convert Model Weights

```bash
# Convert for 16-GPU setup
python convert.py \
  --hf-ckpt-path /path/to/DeepSeek-V3 \
  --save-path /path/to/DeepSeek-V3-Demo \
  --n-experts 256 \
  --model-parallel 16
```

### Run Interactive Chat

```bash
# Single node, 8 GPUs
torchrun --nproc-per-node 8 generate.py \
  --ckpt-path /path/to/DeepSeek-V3-Demo \
  --config configs/config_671B.json \
  --interactive \
  --temperature 0.7 \
  --max-new-tokens 200
```

---

## Basic Usage Examples

### 1. Simple Text Generation

```python
import torch
import json
from model import Transformer, ModelArgs
from generate import generate
from transformers import AutoTokenizer
from safetensors.torch import load_model

# Load configuration
with open("configs/config_671B.json") as f:
    config = json.load(f)
    args = ModelArgs(**config)

# Initialize model
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
model = Transformer(args)

# Load weights
load_model(model, "/path/to/checkpoint/model0-mp1.safetensors")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# Generate text
prompt = "Explain quantum computing in simple terms:"
prompt_tokens = [tokenizer.encode(prompt)]

completions = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=150,
    eos_id=tokenizer.eos_token_id,
    temperature=0.8
)

# Decode and print result
completion_text = tokenizer.decode(completions[0], skip_special_tokens=True)
print(f"Generated text: {completion_text}")
```

### 2. Batch Processing

```python
import torch
from generate import generate
from transformers import AutoTokenizer

# Load model and tokenizer (as above)
# ... model initialization code ...

# Prepare multiple prompts
prompts = [
    "What is machine learning?",
    "Explain the theory of relativity.",
    "How do neural networks work?",
    "What is the future of AI?"
]

# Tokenize all prompts
prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# Generate completions for all prompts
completions = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=100,
    eos_id=tokenizer.eos_token_id,
    temperature=0.7
)

# Process results
for i, (prompt, completion) in enumerate(zip(prompts, completions)):
    completion_text = tokenizer.decode(completion, skip_special_tokens=True)
    print(f"\nPrompt {i+1}: {prompt}")
    print(f"Response: {completion_text}")
    print("-" * 50)
```

### 3. Chat Interface Implementation

```python
from transformers import AutoTokenizer
from generate import generate

class DeepSeekChat:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
    
    def chat(self, user_input: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Process user input and generate response.
        
        Args:
            user_input (str): User's message
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Model's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Apply chat template
        prompt_tokens = self.tokenizer.apply_chat_template(
            self.conversation_history, 
            add_generation_prompt=True
        )
        
        # Generate response
        completion_tokens = generate(
            model=self.model,
            prompt_tokens=[prompt_tokens],
            max_new_tokens=max_new_tokens,
            eos_id=self.tokenizer.eos_token_id,
            temperature=temperature
        )
        
        # Decode response
        response = self.tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()

# Usage example
chat = DeepSeekChat(model, tokenizer)

# Interactive chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['/exit', '/quit']:
        break
    elif user_input.lower() == '/clear':
        chat.clear_history()
        print("Conversation history cleared.")
        continue
    
    response = chat.chat(user_input)
    print(f"DeepSeek: {response}")
```

---

## Advanced Usage Scenarios

### 1. Custom Model Configuration

```python
from model import ModelArgs, Transformer

# Create custom smaller model for experimentation
custom_args = ModelArgs(
    vocab_size=50000,
    dim=1024,
    inter_dim=2816,
    moe_inter_dim=512,
    n_layers=12,
    n_dense_layers=2,
    n_heads=16,
    n_routed_experts=32,
    n_shared_experts=1,
    n_activated_experts=4,
    max_seq_len=2048,
    dtype="bf16"
)

# Initialize custom model
model = Transformer(custom_args)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. Fine-tuning Setup

```python
import torch
from torch.optim import AdamW
from model import Transformer, ModelArgs

# Load pre-trained model
args = ModelArgs(**config)
model = Transformer(args)
load_model(model, checkpoint_path)

# Set up for fine-tuning
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Example training step
def training_step(batch_tokens, batch_labels):
    """
    Perform a single training step.
    
    Args:
        batch_tokens (torch.Tensor): Input token IDs
        batch_labels (torch.Tensor): Target token IDs
        
    Returns:
        float: Training loss
    """
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(batch_tokens[:, :-1])  # Exclude last token from input
    targets = batch_labels[:, 1:]  # Exclude first token from targets
    
    # Compute loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Training loop example
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = training_step(batch['input_ids'], batch['labels'])
        print(f"Loss: {loss:.4f}")
```

### 3. Model Evaluation

```python
import torch
from torch.utils.data import DataLoader
from generate import generate

def evaluate_model(model, tokenizer, eval_dataset, max_new_tokens=50):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer instance
        eval_dataset: Evaluation dataset
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_samples = 0
    total_length = 0
    
    with torch.no_grad():
        for batch in eval_dataset:
            prompts = batch['prompts']
            
            # Tokenize prompts
            prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
            
            # Generate completions
            completions = generate(
                model=model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
                temperature=0.0  # Greedy decoding for evaluation
            )
            
            # Accumulate statistics
            total_samples += len(completions)
            total_length += sum(len(comp) for comp in completions)
    
    avg_length = total_length / total_samples
    return {"average_completion_length": avg_length, "total_samples": total_samples}
```

---

## Distributed Computing

### 1. Multi-GPU Setup

```python
import os
import torch
import torch.distributed as dist
from model import Transformer, ModelArgs

def setup_distributed():
    """Initialize distributed environment."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl")
    
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank

def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

# Usage
world_size, rank, local_rank = setup_distributed()

try:
    # Initialize model with distributed support
    model = Transformer(args)
    
    # Load appropriate shard
    checkpoint_path = f"/path/to/checkpoint/model{rank}-mp{world_size}.safetensors"
    load_model(model, checkpoint_path)
    
    # Run inference
    # ... inference code ...
    
finally:
    cleanup_distributed()
```

### 2. Multi-Node Inference

```bash
#!/bin/bash
# launch_multinode.sh

# Node 0 (master)
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank=0 \
  --master-addr=192.168.1.100 \
  --master-port=29500 \
  generate.py \
  --ckpt-path /path/to/checkpoint \
  --config configs/config_671B.json \
  --interactive

# Node 1 (worker) - run on second machine
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank=1 \
  --master-addr=192.168.1.100 \
  --master-port=29500 \
  generate.py \
  --ckpt-path /path/to/checkpoint \
  --config configs/config_671B.json \
  --interactive
```

---

## Model Conversion Workflows

### 1. Hugging Face to DeepSeek Format

```python
import os
from convert import main as convert_main

def convert_hf_model(hf_path, output_path, model_parallel=8):
    """
    Convert Hugging Face model to DeepSeek format.
    
    Args:
        hf_path (str): Path to Hugging Face model
        output_path (str): Output directory
        model_parallel (int): Number of parallel shards
    """
    
    # Determine number of experts from config
    config_path = os.path.join(hf_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    n_experts = config.get("n_routed_experts", 256)
    
    # Convert model
    convert_main(
        hf_ckpt_path=hf_path,
        save_path=output_path,
        n_experts=n_experts,
        mp=model_parallel
    )
    
    print(f"Model converted successfully to {output_path}")

# Usage
convert_hf_model(
    hf_path="/path/to/huggingface/model",
    output_path="/path/to/converted/model",
    model_parallel=16
)
```

### 2. FP8 to BF16 Conversion

```python
from fp8_cast_bf16 import main as fp8_to_bf16

def convert_fp8_to_bf16(fp8_path, bf16_path):
    """
    Convert FP8 weights to BF16 format.
    
    Args:
        fp8_path (str): Path to FP8 model weights
        bf16_path (str): Output path for BF16 weights
    """
    
    try:
        fp8_to_bf16(fp8_path, bf16_path)
        print(f"Successfully converted FP8 model to BF16 at {bf16_path}")
        
        # Verify conversion
        import os
        if os.path.exists(os.path.join(bf16_path, "model.safetensors.index.json")):
            print("Model index file created successfully")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise

# Usage
convert_fp8_to_bf16(
    fp8_path="/path/to/fp8/model",
    bf16_path="/path/to/bf16/model"
)
```

---

## Performance Optimization

### 1. Memory Optimization

```python
import torch
from model import Transformer, ModelArgs

def optimize_memory_usage(args: ModelArgs):
    """
    Configure model for optimal memory usage.
    
    Args:
        args (ModelArgs): Model configuration
        
    Returns:
        ModelArgs: Optimized configuration
    """
    
    # Reduce batch size and sequence length for memory constraints
    args.max_batch_size = min(args.max_batch_size, 4)
    args.max_seq_len = min(args.max_seq_len, 8192)
    
    # Use FP8 for memory efficiency
    args.dtype = "fp8"
    
    return args

def setup_memory_efficient_model(config_path: str):
    """Setup model with memory optimizations."""
    
    # Load and optimize config
    with open(config_path) as f:
        config = json.load(f)
    args = ModelArgs(**config)
    args = optimize_memory_usage(args)
    
    # Enable memory optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize model
    model = Transformer(args)
    
    # Compile model for better performance (PyTorch 2.0+)
    model = torch.compile(model, mode="reduce-overhead")
    
    return model, args

# Usage
model, args = setup_memory_efficient_model("configs/config_671B.json")
```

### 2. Inference Speed Optimization

```python
import time
import torch
from contextlib import contextmanager

@contextmanager
def inference_mode():
    """Context manager for optimized inference."""
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Disable gradient computation and enable mixed precision
            yield

def benchmark_generation(model, tokenizer, prompts, iterations=10):
    """
    Benchmark text generation performance.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer instance
        prompts: List of input prompts
        iterations: Number of benchmark iterations
        
    Returns:
        dict: Performance metrics
    """
    
    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    
    # Warmup
    with inference_mode():
        for _ in range(3):
            generate(model, prompt_tokens[:1], 10, tokenizer.eos_token_id, 0.0)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with inference_mode():
        for _ in range(iterations):
            completions = generate(
                model=model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=100,
                eos_id=tokenizer.eos_token_id,
                temperature=0.0
            )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    tokens_per_second = (len(prompts) * 100) / avg_time
    
    return {
        "total_time": total_time,
        "average_time_per_iteration": avg_time,
        "tokens_per_second": tokens_per_second,
        "batches_processed": len(prompts) * iterations
    }

# Usage
prompts = ["Tell me about AI"] * 4  # Batch of 4 identical prompts
metrics = benchmark_generation(model, tokenizer, prompts)
print(f"Performance: {metrics['tokens_per_second']:.2f} tokens/second")
```

---

## Integration Examples

### 1. Web API Integration

```python
from flask import Flask, request, jsonify
from generate import generate
import threading
import queue

app = Flask(__name__)

# Global model and tokenizer (loaded once)
model = None
tokenizer = None
generation_queue = queue.Queue()

def load_model_once():
    """Load model and tokenizer once at startup."""
    global model, tokenizer
    
    # Load configuration and model
    with open("configs/config_671B.json") as f:
        args = ModelArgs(**json.load(f))
    
    model = Transformer(args)
    load_model(model, "/path/to/checkpoint/model0-mp1.safetensors")
    
    tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")
    print("Model loaded successfully")

@app.route('/generate', methods=['POST'])
def api_generate():
    """API endpoint for text generation."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_new_tokens = data.get('max_new_tokens', 100)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Tokenize prompt
        prompt_tokens = [tokenizer.encode(prompt)]
        
        # Generate completion
        completions = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_token_id,
            temperature=temperature
        )
        
        # Decode result
        completion_text = tokenizer.decode(completions[0], skip_special_tokens=True)
        
        return jsonify({
            "prompt": prompt,
            "completion": completion_text,
            "tokens_generated": len(completions[0])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    load_model_once()
    app.run(host='0.0.0.0', port=8000, threaded=True)
```

### 2. Streaming Generation

```python
import torch
from typing import Iterator
from generate import sample

def stream_generate(
    model: Transformer,
    prompt_tokens: List[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> Iterator[str]:
    """
    Stream tokens as they are generated.
    
    Args:
        model: Transformer model
        prompt_tokens: Tokenized input prompt
        max_new_tokens: Maximum tokens to generate
        eos_id: End-of-sequence token ID
        temperature: Sampling temperature
        
    Yields:
        str: Decoded tokens as they are generated
    """
    
    tokens = torch.tensor([prompt_tokens], device="cuda")
    prompt_len = len(prompt_tokens)
    
    with torch.inference_mode():
        for i in range(max_new_tokens):
            # Forward pass
            logits = model(tokens, start_pos=0)
            
            # Sample next token
            if temperature > 0:
                next_token = sample(logits, temperature)
            else:
                next_token = logits.argmax(dim=-1)
            
            # Check for end of sequence
            if next_token.item() == eos_id:
                break
            
            # Append token and yield decoded text
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Decode only the new token
            new_token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            yield new_token_text

# Usage example
def demo_streaming():
    """Demonstrate streaming generation."""
    prompt = "The future of artificial intelligence"
    prompt_tokens = tokenizer.encode(prompt)
    
    print(f"Prompt: {prompt}")
    print("Generated text: ", end="", flush=True)
    
    for token_text in stream_generate(model, prompt_tokens, 100, tokenizer.eos_token_id, 0.7):
        print(token_text, end="", flush=True)
    
    print("\n[Generation complete]")
```

### 3. Custom Sampling Strategies

```python
import torch
import torch.nn.functional as F

def top_k_top_p_sampling(logits, k=50, p=0.9, temperature=1.0):
    """
    Advanced sampling with top-k and top-p filtering.
    
    Args:
        logits (torch.Tensor): Model logits
        k (int): Top-k filtering parameter
        p (float): Top-p (nucleus) filtering parameter
        temperature (float): Temperature scaling
        
    Returns:
        torch.Tensor: Sampled token
    """
    
    # Apply temperature
    logits = logits / max(temperature, 1e-5)
    
    # Top-k filtering
    if k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, top_k_indices, top_k_logits)
    
    # Top-p (nucleus) filtering
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

def generate_with_custom_sampling(model, prompt_tokens, max_new_tokens, eos_id, **sampling_kwargs):
    """Generate text with custom sampling strategy."""
    
    tokens = torch.tensor([prompt_tokens], device="cuda")
    generated_tokens = []
    
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(tokens, start_pos=0)
            
            # Use custom sampling
            next_token = top_k_top_p_sampling(logits, **sampling_kwargs)
            
            if next_token.item() == eos_id:
                break
            
            generated_tokens.append(next_token.item())
            tokens = torch.cat([tokens, next_token], dim=1)
    
    return generated_tokens

# Usage
completion = generate_with_custom_sampling(
    model=model,
    prompt_tokens=tokenizer.encode("Write a story about"),
    max_new_tokens=200,
    eos_id=tokenizer.eos_token_id,
    k=40,
    p=0.95,
    temperature=0.8
)
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
args.max_batch_size = 1

# Reduce sequence length
args.max_seq_len = 4096

# Use gradient checkpointing (for training)
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()
```

#### 2. Distributed Setup Issues

**Problem**: Distributed initialization fails

**Solutions**:
```bash
# Check environment variables
echo $WORLD_SIZE $RANK $LOCAL_RANK $MASTER_ADDR $MASTER_PORT

# Verify network connectivity
ping $MASTER_ADDR

# Check NCCL setup
python -c "import torch; torch.distributed.init_process_group('nccl')"
```

#### 3. Model Loading Errors

**Problem**: Cannot load model weights

**Solutions**:
```python
import os
from safetensors.torch import safe_open

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint file integrity."""
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"Checkpoint contains {len(keys)} parameters")
            print(f"Sample keys: {keys[:5]}")
        return True
    except Exception as e:
        print(f"Checkpoint verification failed: {e}")
        return False

# Usage
verify_checkpoint("/path/to/model0-mp1.safetensors")
```

#### 4. Performance Issues

**Problem**: Slow inference speed

**Solutions**:
```python
# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode="max-autotune")

# Optimize data loading
torch.set_num_threads(8)

# Use appropriate precision
torch.set_default_dtype(torch.bfloat16)
```

### Debugging Utilities

```python
def debug_model_state(model):
    """Print model state for debugging."""
    
    print("Model Configuration:")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Check for NaN or inf values
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN values in {name}")
        if torch.isinf(param).any():
            print(f"WARNING: Inf values in {name}")

def debug_generation(model, tokenizer, prompt, debug=True):
    """Debug text generation step by step."""
    
    prompt_tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_tokens], device="cuda")
    
    if debug:
        print(f"Prompt: {prompt}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Token tensor shape: {tokens.shape}")
    
    with torch.inference_mode():
        # Forward pass
        logits = model(tokens, start_pos=0)
        
        if debug:
            print(f"Logits shape: {logits.shape}")
            print(f"Logits dtype: {logits.dtype}")
            print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Sample token
        next_token = logits.argmax(dim=-1)
        next_token_text = tokenizer.decode([next_token.item()])
        
        if debug:
            print(f"Next token ID: {next_token.item()}")
            print(f"Next token text: '{next_token_text}'")
        
        return next_token_text

# Usage
debug_generation(model, tokenizer, "Hello, world!")
```

---

## Advanced Configuration Examples

### 1. Custom Expert Configuration

```python
from model import ModelArgs

def create_custom_moe_config():
    """Create custom MoE configuration for specific use cases."""
    
    # High-capacity configuration
    high_capacity_args = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_routed_experts=128,
        n_shared_experts=2,
        n_activated_experts=8,
        n_expert_groups=4,
        n_limited_groups=2,
        route_scale=2.0,
        score_func="sigmoid"
    )
    
    # Memory-efficient configuration
    memory_efficient_args = ModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_routed_experts=64,
        n_shared_experts=1,
        n_activated_experts=4,
        n_expert_groups=2,
        n_limited_groups=1,
        route_scale=1.5,
        score_func="softmax"
    )
    
    return high_capacity_args, memory_efficient_args

# Usage
high_cap_args, mem_eff_args = create_custom_moe_config()
```

### 2. Dynamic Configuration Loading

```python
import json
import os
from typing import Dict, Any

class ConfigManager:
    """Manage multiple model configurations."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files."""
        for config_file in os.listdir(self.config_dir):
            if config_file.endswith('.json'):
                config_name = config_file.replace('.json', '')
                config_path = os.path.join(self.config_dir, config_file)
                
                with open(config_path) as f:
                    self.configs[config_name] = json.load(f)
    
    def get_config(self, config_name: str) -> ModelArgs:
        """Get configuration by name."""
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        return ModelArgs(**self.configs[config_name])
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        return list(self.configs.keys())
    
    def create_custom_config(self, base_config: str, modifications: Dict[str, Any]) -> ModelArgs:
        """Create custom configuration based on existing one."""
        base = self.configs[base_config].copy()
        base.update(modifications)
        return ModelArgs(**base)

# Usage
config_manager = ConfigManager()
print("Available configurations:", config_manager.list_configs())

# Load specific configuration
args = config_manager.get_config("config_671B")

# Create custom configuration
custom_args = config_manager.create_custom_config(
    "config_671B",
    {"max_seq_len": 8192, "temperature": 0.5}
)
```

---

## Production Deployment Examples

### 1. Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Expose port for API
EXPOSE 8000

# Run command
CMD ["python3", "api_server.py"]
```

```bash
# Build and run Docker container
docker build -t deepseek-v3 .
docker run --gpus all -p 8000:8000 -v /path/to/checkpoints:/app/checkpoints deepseek-v3
```

### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek-v3-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek-v3
  template:
    metadata:
      labels:
        app: deepseek-v3
    spec:
      containers:
      - name: deepseek-v3
        image: deepseek-v3:latest
        resources:
          limits:
            nvidia.com/gpu: 8
          requests:
            nvidia.com/gpu: 8
        env:
        - name: WORLD_SIZE
          value: "8"
        - name: MASTER_ADDR
          value: "localhost"
        - name: MASTER_PORT
          value: "29500"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: model-storage
          mountPath: /app/checkpoints
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

### 3. Load Balancing and Scaling

```python
import asyncio
import aiohttp
from typing import List

class ModelLoadBalancer:
    """Load balancer for multiple model instances."""
    
    def __init__(self, model_endpoints: List[str]):
        self.endpoints = model_endpoints
        self.current_index = 0
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using round-robin load balancing."""
        
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": prompt, **kwargs}
            
            async with session.post(f"{endpoint}/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["completion"]
                else:
                    raise Exception(f"Request failed with status {response.status}")

# Usage
load_balancer = ModelLoadBalancer([
    "http://gpu-node-1:8000",
    "http://gpu-node-2:8000",
    "http://gpu-node-3:8000",
    "http://gpu-node-4:8000"
])

async def main():
    completion = await load_balancer.generate_text(
        "Explain quantum computing",
        max_new_tokens=150,
        temperature=0.7
    )
    print(completion)

# Run
asyncio.run(main())
```

---

## Monitoring and Logging

### 1. Performance Monitoring

```python
import time
import psutil
import torch
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    generation_time: float
    tokens_per_second: float
    memory_usage_gb: float
    gpu_utilization: float
    
class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self):
        self.metrics_history = []
    
    def measure_generation(self, model, prompt_tokens, max_new_tokens, **kwargs) -> PerformanceMetrics:
        """Measure performance of text generation."""
        
        # Memory before generation
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        
        # Time generation
        start_time = time.time()
        
        completions = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        generation_time = end_time - start_time
        total_tokens = sum(len(comp) for comp in completions)
        tokens_per_second = total_tokens / generation_time
        
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_usage = memory_after - memory_before
        
        # GPU utilization (approximate)
        gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        metrics = PerformanceMetrics(
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            memory_usage_gb=memory_usage,
            gpu_utilization=gpu_utilization
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all measurements."""
        if not self.metrics_history:
            return {}
        
        return {
            "avg_generation_time": sum(m.generation_time for m in self.metrics_history) / len(self.metrics_history),
            "avg_tokens_per_second": sum(m.tokens_per_second for m in self.metrics_history) / len(self.metrics_history),
            "avg_memory_usage_gb": sum(m.memory_usage_gb for m in self.metrics_history) / len(self.metrics_history),
            "avg_gpu_utilization": sum(m.gpu_utilization for m in self.metrics_history) / len(self.metrics_history)
        }

# Usage
monitor = PerformanceMonitor()

# Measure performance
prompt_tokens = [tokenizer.encode("What is the meaning of life?")]
metrics = monitor.measure_generation(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=100,
    eos_id=tokenizer.eos_token_id,
    temperature=0.7
)

print(f"Generation time: {metrics.generation_time:.2f}s")
print(f"Tokens/second: {metrics.tokens_per_second:.2f}")
print(f"Memory usage: {metrics.memory_usage_gb:.2f} GB")
```

### 2. Logging Configuration

```python
import logging
import sys
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup comprehensive logging for the application."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers
    )
    
    # Create application logger
    logger = logging.getLogger("deepseek_v3")
    return logger

# Usage
logger = setup_logging(
    log_level=logging.INFO,
    log_file=f"deepseek_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Log generation events
def logged_generate(model, prompt_tokens, **kwargs):
    """Generate text with comprehensive logging."""
    
    logger.info(f"Starting generation for {len(prompt_tokens)} prompts")
    logger.info(f"Generation parameters: {kwargs}")
    
    try:
        start_time = time.time()
        completions = generate(model=model, prompt_tokens=prompt_tokens, **kwargs)
        generation_time = time.time() - start_time
        
        total_tokens = sum(len(comp) for comp in completions)
        logger.info(f"Generation completed in {generation_time:.2f}s")
        logger.info(f"Generated {total_tokens} tokens ({total_tokens/generation_time:.2f} tokens/s)")
        
        return completions
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
```

---

This comprehensive usage guide provides practical examples for all major use cases of the DeepSeek-V3 model, from basic text generation to advanced production deployments. The examples are designed to be immediately usable and demonstrate best practices for optimal performance and reliability.