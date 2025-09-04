# DeepSeek-V3 API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
   - [ModelArgs](#modelargs)
   - [Transformer](#transformer)
   - [Block](#block)
   - [MLA (Multi-head Latent Attention)](#mla-multi-head-latent-attention)
   - [MoE (Mixture-of-Experts)](#moe-mixture-of-experts)
   - [Linear Layers](#linear-layers)
   - [Utility Components](#utility-components)
3. [Text Generation API](#text-generation-api)
   - [generate()](#generate)
   - [sample()](#sample)
   - [main()](#main-generation)
4. [Quantization and Kernels](#quantization-and-kernels)
   - [act_quant()](#act_quant)
   - [weight_dequant()](#weight_dequant)
   - [fp8_gemm()](#fp8_gemm)
5. [Model Conversion Utilities](#model-conversion-utilities)
   - [convert.py main()](#convert-main)
   - [fp8_cast_bf16.py main()](#fp8_cast_bf16-main)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Installation and Setup](#installation-and-setup)

---

## Overview

DeepSeek-V3 is a state-of-the-art 671B parameter Mixture-of-Experts (MoE) language model with 37B activated parameters per token. The implementation features:

- **Multi-head Latent Attention (MLA)**: Efficient attention mechanism with compressed key-value representations
- **DeepSeekMoE**: Advanced mixture-of-experts architecture with load balancing
- **FP8 Training Support**: Native FP8 mixed precision training for efficiency
- **Distributed Inference**: Multi-GPU and multi-node inference capabilities
- **Rotary Position Embeddings**: Enhanced positional encoding with YARN scaling

---

## Model Architecture

### ModelArgs

**Class**: `ModelArgs`

Configuration dataclass that defines all model hyperparameters and architecture settings.

#### Attributes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_batch_size` | int | 8 | Maximum batch size for inference |
| `max_seq_len` | int | 16384 | Maximum sequence length |
| `dtype` | Literal["bf16", "fp8"] | "bf16" | Data type for computations |
| `vocab_size` | int | 102400 | Vocabulary size |
| `dim` | int | 2048 | Model dimension |
| `inter_dim` | int | 10944 | Intermediate dimension for MLP layers |
| `moe_inter_dim` | int | 1408 | Intermediate dimension for MoE layers |
| `n_layers` | int | 27 | Number of transformer layers |
| `n_dense_layers` | int | 1 | Number of dense (non-MoE) layers |
| `n_heads` | int | 16 | Number of attention heads |
| `n_routed_experts` | int | 64 | Number of routed experts for MoE |
| `n_shared_experts` | int | 2 | Number of shared experts for MoE |
| `n_activated_experts` | int | 6 | Number of activated experts per token |
| `n_expert_groups` | int | 1 | Number of expert groups |
| `n_limited_groups` | int | 1 | Number of limited groups for routing |
| `score_func` | Literal["softmax", "sigmoid"] | "softmax" | Scoring function for expert routing |
| `route_scale` | float | 1.0 | Scaling factor for routing scores |
| `q_lora_rank` | int | 0 | LoRA rank for query projections |
| `kv_lora_rank` | int | 512 | LoRA rank for key-value projections |
| `qk_nope_head_dim` | int | 128 | Dimension for non-positional query-key projections |
| `qk_rope_head_dim` | int | 64 | Dimension for rotary positional query-key projections |
| `v_head_dim` | int | 128 | Dimension for value projections |
| `original_seq_len` | int | 4096 | Original sequence length for YARN scaling |
| `rope_theta` | float | 10000.0 | Base for rotary positional encoding |
| `rope_factor` | float | 40 | Scaling factor for extended sequence lengths |
| `beta_fast` | int | 32 | Fast beta correction factor for YARN |
| `beta_slow` | int | 1 | Slow beta correction factor for YARN |
| `mscale` | float | 1.0 | Scaling factor for extended attention |

#### Example Usage

```python
from model import ModelArgs

# Default configuration for smaller model
args = ModelArgs()

# Configuration for DeepSeek-V3-671B
args = ModelArgs(
    vocab_size=129280,
    dim=7168,
    inter_dim=18432,
    moe_inter_dim=2048,
    n_layers=61,
    n_dense_layers=3,
    n_heads=128,
    n_routed_experts=256,
    n_shared_experts=1,
    n_activated_experts=8,
    dtype="fp8"
)
```

---

### Transformer

**Class**: `Transformer(nn.Module)`

Main transformer model class that orchestrates the entire model architecture.

#### Constructor

```python
def __init__(self, args: ModelArgs)
```

**Parameters:**
- `args` (ModelArgs): Model configuration parameters

#### Methods

##### forward()

```python
@torch.inference_mode()
def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor
```

**Parameters:**
- `tokens` (torch.Tensor): Input token IDs with shape `(batch_size, seq_len)`
- `start_pos` (int, optional): Starting position for rotary embeddings. Default: 0

**Returns:**
- `torch.Tensor`: Logits tensor with shape `(batch_size, vocab_size)`

**Description:**
Performs forward pass through the entire transformer model, including embedding, all transformer blocks, normalization, and output projection.

#### Example Usage

```python
import torch
from model import Transformer, ModelArgs

# Initialize model
args = ModelArgs()
model = Transformer(args)

# Generate logits for input tokens
tokens = torch.randint(0, args.vocab_size, (2, 128))
logits = model(tokens)
print(f"Output shape: {logits.shape}")  # (2, vocab_size)
```

---

### Block

**Class**: `Block(nn.Module)`

Individual transformer block combining attention and feed-forward layers.

#### Constructor

```python
def __init__(self, layer_id: int, args: ModelArgs)
```

**Parameters:**
- `layer_id` (int): Layer index in the transformer (determines if MLP or MoE is used)
- `args` (ModelArgs): Model configuration parameters

#### Methods

##### forward()

```python
def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `start_pos` (int): Starting position in the sequence
- `freqs_cis` (torch.Tensor): Precomputed rotary embedding frequencies
- `mask` (Optional[torch.Tensor]): Attention mask

**Returns:**
- `torch.Tensor`: Output tensor after block computation

---

### MLA (Multi-head Latent Attention)

**Class**: `MLA(nn.Module)`

Advanced attention mechanism with compressed key-value representations for improved efficiency.

#### Constructor

```python
def __init__(self, args: ModelArgs)
```

#### Key Features

- **Low-rank key-value compression**: Reduces memory usage for long sequences
- **Rotary position embeddings**: Enhanced positional encoding
- **Distributed attention**: Support for multi-GPU attention computation
- **Two implementation modes**: "naive" and "absorb" for different memory-computation tradeoffs

#### Methods

##### forward()

```python
def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor with shape `(batch_size, seq_len, dim)`
- `start_pos` (int): Starting position for caching
- `freqs_cis` (torch.Tensor): Precomputed rotary embedding frequencies
- `mask` (Optional[torch.Tensor]): Attention mask to exclude certain positions

**Returns:**
- `torch.Tensor`: Output tensor with the same shape as input

#### Example Usage

```python
from model import MLA, ModelArgs

args = ModelArgs()
mla = MLA(args)

# Forward pass
batch_size, seq_len = 2, 128
x = torch.randn(batch_size, seq_len, args.dim)
freqs_cis = torch.randn(seq_len, args.qk_rope_head_dim // 2)
output = mla(x, start_pos=0, freqs_cis=freqs_cis, mask=None)
```

---

### MoE (Mixture-of-Experts)

**Class**: `MoE(nn.Module)`

Mixture-of-Experts module that routes inputs to specialized expert networks.

#### Constructor

```python
def __init__(self, args: ModelArgs)
```

#### Key Features

- **Expert routing**: Intelligent routing of inputs to the most relevant experts
- **Load balancing**: Auxiliary-loss-free load balancing strategy
- **Shared experts**: Always-activated experts for common computations
- **Distributed experts**: Support for distributing experts across multiple GPUs

#### Methods

##### forward()

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Output tensor after expert routing and computation

#### Example Usage

```python
from model import MoE, ModelArgs

args = ModelArgs()
moe = MoE(args)

# Forward pass
x = torch.randn(2, 128, args.dim)
output = moe(x)
```

---

### Linear Layers

The model implements several specialized linear layer variants for distributed computing:

#### Linear

**Class**: `Linear(nn.Module)`

Base linear layer with quantization support.

```python
def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None)
```

#### ColumnParallelLinear

**Class**: `ColumnParallelLinear(Linear)`

Linear layer with column parallelism, splitting output features across processes.

```python
def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None)
```

#### RowParallelLinear

**Class**: `RowParallelLinear(Linear)`

Linear layer with row parallelism, splitting input features across processes.

```python
def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None)
```

#### Example Usage

```python
from model import ColumnParallelLinear, RowParallelLinear

# Column parallel layer (splits output across GPUs)
col_layer = ColumnParallelLinear(in_features=1024, out_features=4096)

# Row parallel layer (splits input across GPUs)  
row_layer = RowParallelLinear(in_features=4096, out_features=1024)

x = torch.randn(2, 128, 1024)
y = col_layer(x)  # Shape: (2, 128, 4096//world_size)
z = row_layer(y)  # Shape: (2, 128, 1024)
```

---

### Utility Components

#### RMSNorm

**Class**: `RMSNorm(nn.Module)`

Root Mean Square Layer Normalization for improved training stability.

```python
def __init__(self, dim: int, eps: float = 1e-6)
```

**Parameters:**
- `dim` (int): Dimension of the input tensor
- `eps` (float): Epsilon value for numerical stability

#### ParallelEmbedding

**Class**: `ParallelEmbedding(nn.Module)`

Embedding layer with distributed parallelism support.

```python
def __init__(self, vocab_size: int, dim: int)
```

**Parameters:**
- `vocab_size` (int): Vocabulary size
- `dim` (int): Embedding dimension

#### Rotary Position Embeddings

##### precompute_freqs_cis()

```python
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor
```

Precomputes frequency-based complex exponential values for rotary positional embeddings with YARN scaling support.

**Parameters:**
- `args` (ModelArgs): Model arguments containing positional embedding parameters

**Returns:**
- `torch.Tensor`: Precomputed complex exponential values

##### apply_rotary_emb()

```python
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor
```

Applies rotary positional embeddings to input tensor.

**Parameters:**
- `x` (torch.Tensor): Input tensor with positional embeddings to be applied
- `freqs_cis` (torch.Tensor): Precomputed complex exponential values

**Returns:**
- `torch.Tensor`: Tensor with rotary embeddings applied

---

## Text Generation API

### generate()

**Function**: `generate(model, prompt_tokens, max_new_tokens, eos_id, temperature=1.0)`

Main text generation function that performs autoregressive token generation.

```python
@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]
```

**Parameters:**
- `model` (Transformer): The transformer model for token generation
- `prompt_tokens` (List[List[int]]): List of tokenized prompts for each sequence
- `max_new_tokens` (int): Maximum number of new tokens to generate
- `eos_id` (int): End-of-sequence token ID
- `temperature` (float, optional): Temperature for sampling. Default: 1.0

**Returns:**
- `List[List[int]]`: Generated completion tokens for each input sequence

**Features:**
- Batch processing support
- Temperature-controlled sampling
- Early stopping on EOS token
- KV-cache optimization for efficient generation

#### Example Usage

```python
from generate import generate
from transformers import AutoTokenizer

# Load model and tokenizer
model = Transformer(args)
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# Prepare prompts
prompts = ["Hello, how are you?", "What is the capital of France?"]
prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# Generate completions
completions = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=100,
    eos_id=tokenizer.eos_token_id,
    temperature=0.7
)

# Decode results
for i, completion in enumerate(completions):
    text = tokenizer.decode(completion, skip_special_tokens=True)
    print(f"Completion {i}: {text}")
```

---

### sample()

**Function**: `sample(logits, temperature=1.0)`

Samples tokens from logits using temperature scaling.

```python
def sample(logits, temperature: float = 1.0) -> torch.Tensor
```

**Parameters:**
- `logits` (torch.Tensor): Logits tensor for token predictions
- `temperature` (float, optional): Temperature for scaling logits. Default: 1.0

**Returns:**
- `torch.Tensor`: Sampled token

**Description:**
Uses the Gumbel-max trick for efficient sampling with temperature control.

---

### main() (Generation)

**Function**: `main(ckpt_path, config, input_file="", interactive=True, max_new_tokens=100, temperature=1.0)`

Main entry point for text generation with support for interactive chat and batch processing.

```python
def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None
```

**Parameters:**
- `ckpt_path` (str): Path to model checkpoint directory
- `config` (str): Path to model configuration file
- `input_file` (str, optional): Path to file containing input prompts
- `interactive` (bool, optional): Whether to run in interactive mode. Default: True
- `max_new_tokens` (int, optional): Maximum tokens to generate. Default: 100
- `temperature` (float, optional): Sampling temperature. Default: 1.0

**Features:**
- Interactive chat interface with conversation history
- Batch processing from input files
- Distributed inference support
- Chat template formatting

#### Command Line Usage

```bash
# Interactive mode
torchrun --nnodes 2 --nproc-per-node 8 generate.py \
  --ckpt-path /path/to/checkpoint \
  --config configs/config_671B.json \
  --interactive \
  --temperature 0.7 \
  --max-new-tokens 200

# Batch processing
torchrun --nnodes 2 --nproc-per-node 8 generate.py \
  --ckpt-path /path/to/checkpoint \
  --config configs/config_671B.json \
  --input-file prompts.txt \
  --max-new-tokens 100
```

---

## Quantization and Kernels

### act_quant()

**Function**: `act_quant(x, block_size=128)`

Quantizes activation tensors using block-wise FP8 quantization.

```python
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]
```

**Parameters:**
- `x` (torch.Tensor): Input tensor to quantize (must be contiguous)
- `block_size` (int, optional): Block size for quantization. Default: 128

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: Quantized tensor (FP8) and scaling factors (FP32)

**Requirements:**
- Input tensor must be contiguous
- Last dimension size must be divisible by `block_size`

#### Example Usage

```python
from kernel import act_quant

x = torch.randn(2, 128, 1024).contiguous()
quantized_x, scales = act_quant(x, block_size=128)
print(f"Original dtype: {x.dtype}, Quantized dtype: {quantized_x.dtype}")
```

---

### weight_dequant()

**Function**: `weight_dequant(x, s, block_size=128)`

Dequantizes weight tensors using provided scaling factors.

```python
def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Quantized weight tensor with shape `(M, N)`
- `s` (torch.Tensor): Scale tensor with shape `(M, N)`
- `block_size` (int, optional): Block size for dequantization. Default: 128

**Returns:**
- `torch.Tensor`: Dequantized weight tensor

**Requirements:**
- Both input tensors must be contiguous and 2-dimensional

---

### fp8_gemm()

**Function**: `fp8_gemm(a, a_s, b, b_s)`

Performs matrix multiplication using FP8 precision with scaling factors.

```python
def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `a` (torch.Tensor): First input matrix (must be contiguous)
- `a_s` (torch.Tensor): Scaling factor for first matrix (must be contiguous)
- `b` (torch.Tensor): Second input matrix (must be contiguous)
- `b_s` (torch.Tensor): Scaling factor for second matrix (must be contiguous)

**Returns:**
- `torch.Tensor`: Result of matrix multiplication

**Description:**
High-performance FP8 GEMM implementation using Triton kernels with automatic tuning for optimal performance.

---

## Model Conversion Utilities

### convert.py main()

**Function**: `main(hf_ckpt_path, save_path, n_experts, mp)`

Converts Hugging Face model checkpoints to the DeepSeek-V3 format with model parallelism support.

```python
def main(hf_ckpt_path, save_path, n_experts, mp)
```

**Parameters:**
- `hf_ckpt_path` (str): Path to Hugging Face checkpoint directory
- `save_path` (str): Output directory for converted checkpoints
- `n_experts` (int): Total number of experts in the model
- `mp` (int): Model parallelism factor

**Features:**
- Converts parameter names from Hugging Face format
- Splits parameters across multiple GPUs for model parallelism
- Handles expert routing for MoE models
- Copies tokenizer files

#### Command Line Usage

```bash
python convert.py \
  --hf-ckpt-path /path/to/huggingface/checkpoint \
  --save-path /path/to/converted/checkpoint \
  --n-experts 256 \
  --model-parallel 16
```

---

### fp8_cast_bf16.py main()

**Function**: `main(fp8_path, bf16_path)`

Converts FP8 quantized weights to BF16 format.

```python
def main(fp8_path, bf16_path)
```

**Parameters:**
- `fp8_path` (str): Path to directory containing FP8 weights
- `bf16_path` (str): Output directory for BF16 weights

**Features:**
- Batch processing of safetensor files
- Memory-efficient conversion with caching
- Automatic model index file updates
- Error handling for missing scale tensors

#### Command Line Usage

```bash
python fp8_cast_bf16.py \
  --input-fp8-hf-path /path/to/fp8/weights \
  --output-bf16-hf-path /path/to/bf16/weights
```

---

## Configuration

### Model Configurations

The project includes three pre-defined configurations:

#### config_16B.json
- Small model configuration for testing
- 16B total parameters

#### config_236B.json  
- Medium model configuration
- 236B total parameters

#### config_671B.json
- Full DeepSeek-V3 configuration
- 671B total parameters with 37B activated

### Example Configuration

```json
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}
```

---

## Usage Examples

### Basic Model Inference

```python
import torch
import json
from model import Transformer, ModelArgs
from generate import generate
from transformers import AutoTokenizer

# Load configuration
with open("configs/config_671B.json") as f:
    args = ModelArgs(**json.load(f))

# Initialize model
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
model = Transformer(args)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")

# Generate text
prompt = "What is artificial intelligence?"
prompt_tokens = [tokenizer.encode(prompt)]
completions = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=100,
    eos_id=tokenizer.eos_token_id,
    temperature=0.7
)

completion_text = tokenizer.decode(completions[0], skip_special_tokens=True)
print(f"Generated: {completion_text}")
```

### Distributed Inference

```python
import os
import torch.distributed as dist

# Initialize distributed environment
world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))

if world_size > 1:
    dist.init_process_group("nccl")

torch.cuda.set_device(local_rank)

# Load model with distributed support
model = Transformer(args)
# ... load weights and run inference
```

### Custom Model Configuration

```python
from model import ModelArgs, Transformer

# Create custom configuration
custom_args = ModelArgs(
    vocab_size=50000,
    dim=1024,
    n_layers=12,
    n_heads=16,
    max_seq_len=2048,
    dtype="bf16"
)

# Initialize model with custom configuration
model = Transformer(custom_args)
```

---

## Installation and Setup

### Dependencies

Install required dependencies:

```bash
pip install torch==2.4.1 triton==3.0.0 transformers==4.46.3 safetensors==0.4.5
```

### System Requirements

- **OS**: Linux with Python 3.10 (Mac and Windows not supported)
- **GPU**: NVIDIA GPUs with CUDA support
- **Memory**: Sufficient GPU memory for model size (varies by configuration)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-V3.git
   cd DeepSeek-V3/inference
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   Download from Hugging Face and place in `/path/to/DeepSeek-V3`

4. **Convert weights:**
   ```bash
   python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 \
     --save-path /path/to/converted \
     --n-experts 256 --model-parallel 16
   ```

5. **Run inference:**
   ```bash
   torchrun --nnodes 2 --nproc-per-node 8 generate.py \
     --ckpt-path /path/to/converted \
     --config configs/config_671B.json \
     --interactive
   ```

---

## Advanced Features

### FP8 Quantization

The model natively supports FP8 quantization for improved memory efficiency and training speed:

- **Activation Quantization**: Block-wise quantization of activations
- **Weight Quantization**: Quantized weight storage with scaling factors
- **Mixed Precision**: Seamless switching between FP8 and BF16 computations

### Multi-Token Prediction (MTP)

The model supports multi-token prediction for improved training efficiency and speculative decoding capabilities.

### Load Balancing

DeepSeek-V3 implements an auxiliary-loss-free load balancing strategy that minimizes performance degradation while ensuring efficient expert utilization.

---

## Error Handling and Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Distributed Setup**: Ensure proper environment variables (WORLD_SIZE, RANK, LOCAL_RANK)
3. **Quantization Errors**: Verify tensor contiguity and dimension requirements

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1
```

---

## Performance Optimization

### Memory Optimization

- Use FP8 quantization for reduced memory usage
- Enable gradient checkpointing for training
- Optimize batch size and sequence length

### Compute Optimization

- Use appropriate model parallelism settings
- Enable Triton kernel optimizations
- Configure optimal block sizes for quantization

---

This documentation provides comprehensive coverage of all public APIs, functions, and components in the DeepSeek-V3 codebase. For additional details, refer to the source code and paper documentation.