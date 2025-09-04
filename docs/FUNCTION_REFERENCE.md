# DeepSeek-V3 Function Reference Guide

## 📑 Quick Function Lookup

This reference provides a comprehensive index of all public functions, methods, and APIs in the DeepSeek-V3 codebase, organized by category for quick lookup.

---

## 🏗️ Model Architecture Functions

### Core Model Classes

| Class | Module | Purpose | Documentation |
|-------|--------|---------|---------------|
| `ModelArgs` | `model.py` | Configuration dataclass | [API Docs](./API_DOCUMENTATION.md#modelargs) |
| `Transformer` | `model.py` | Main transformer model | [API Docs](./API_DOCUMENTATION.md#transformer) |
| `Block` | `model.py` | Transformer block | [API Docs](./API_DOCUMENTATION.md#block) |
| `MLA` | `model.py` | Multi-head Latent Attention | [API Docs](./API_DOCUMENTATION.md#mla-multi-head-latent-attention) |
| `MoE` | `model.py` | Mixture-of-Experts | [API Docs](./API_DOCUMENTATION.md#moe-mixture-of-experts) |
| `MLP` | `model.py` | Multi-Layer Perceptron | [API Docs](./API_DOCUMENTATION.md#utility-components) |
| `Gate` | `model.py` | MoE gating mechanism | [API Docs](./API_DOCUMENTATION.md#moe-mixture-of-experts) |
| `Expert` | `model.py` | Individual expert network | [API Docs](./API_DOCUMENTATION.md#moe-mixture-of-experts) |

### Linear Layer Variants

| Class | Module | Purpose | Signature |
|-------|--------|---------|-----------|
| `Linear` | `model.py` | Base linear layer with quantization | `__init__(in_features, out_features, bias=False, dtype=None)` |
| `ColumnParallelLinear` | `model.py` | Column-parallel linear layer | `__init__(in_features, out_features, bias=False, dtype=None)` |
| `RowParallelLinear` | `model.py` | Row-parallel linear layer | `__init__(in_features, out_features, bias=False, dtype=None)` |
| `ParallelEmbedding` | `model.py` | Parallel embedding layer | `__init__(vocab_size, dim)` |

### Normalization and Utilities

| Function/Class | Module | Purpose | Signature |
|----------------|--------|---------|-----------|
| `RMSNorm` | `model.py` | RMS normalization | `__init__(dim, eps=1e-6)` |
| `linear()` | `model.py` | Quantization-aware linear function | `linear(x, weight, bias=None)` |
| `precompute_freqs_cis()` | `model.py` | Precompute rotary embeddings | `precompute_freqs_cis(args)` |
| `apply_rotary_emb()` | `model.py` | Apply rotary embeddings | `apply_rotary_emb(x, freqs_cis)` |

---

## 🎯 Text Generation Functions

### Core Generation API

| Function | Module | Purpose | Signature |
|----------|--------|---------|-----------|
| `generate()` | `generate.py` | Main text generation function | `generate(model, prompt_tokens, max_new_tokens, eos_id, temperature=1.0)` |
| `sample()` | `generate.py` | Token sampling with temperature | `sample(logits, temperature=1.0)` |
| `main()` | `generate.py` | CLI interface for generation | `main(ckpt_path, config, input_file="", interactive=True, ...)` |

### Function Details

#### generate()
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

**Quick Example:**
```python
completions = generate(model, [tokenizer.encode("Hello")], 50, tokenizer.eos_token_id, 0.7)
```

#### sample()
```python
def sample(logits, temperature: float = 1.0) -> torch.Tensor
```

**Quick Example:**
```python
next_token = sample(model_logits, temperature=0.8)
```

---

## ⚡ Quantization and Kernel Functions

### Quantization Functions

| Function | Module | Purpose | Signature |
|----------|--------|---------|-----------|
| `act_quant()` | `kernel.py` | Quantize activations to FP8 | `act_quant(x, block_size=128)` |
| `weight_dequant()` | `kernel.py` | Dequantize weights from FP8 | `weight_dequant(x, s, block_size=128)` |
| `fp8_gemm()` | `kernel.py` | FP8 matrix multiplication | `fp8_gemm(a, a_s, b, b_s)` |

### Triton Kernels

| Kernel | Module | Purpose | Signature |
|--------|--------|---------|-----------|
| `act_quant_kernel()` | `kernel.py` | GPU activation quantization | `@triton.jit` kernel |
| `weight_dequant_kernel()` | `kernel.py` | GPU weight dequantization | `@triton.jit` kernel |
| `fp8_gemm_kernel()` | `kernel.py` | GPU FP8 matrix multiplication | `@triton.jit` kernel |

### Function Details

#### act_quant()
```python
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]
```

**Quick Example:**
```python
x_quant, scales = act_quant(activations.contiguous(), block_size=128)
```

#### fp8_gemm()
```python
def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor
```

**Quick Example:**
```python
result = fp8_gemm(a_quant, a_scales, b_quant, b_scales)
```

---

## 🔄 Conversion Functions

### Model Format Conversion

| Function | Module | Purpose | Signature |
|----------|--------|---------|-----------|
| `main()` | `convert.py` | HF to DeepSeek conversion | `main(hf_ckpt_path, save_path, n_experts, mp)` |
| `main()` | `fp8_cast_bf16.py` | FP8 to BF16 conversion | `main(fp8_path, bf16_path)` |

### Helper Functions

| Function | Module | Purpose | Description |
|----------|--------|---------|-------------|
| `get_tensor()` | `fp8_cast_bf16.py` | Cached tensor retrieval | Retrieves tensors with intelligent caching |
| Parameter mapping | `convert.py` | Name translation | Maps HF parameter names to DeepSeek format |

### Function Details

#### convert.py main()
```python
def main(hf_ckpt_path, save_path, n_experts, mp)
```

**Quick Example:**
```bash
python convert.py --hf-ckpt-path /path/to/hf --save-path /path/to/output --n-experts 256 --model-parallel 16
```

#### fp8_cast_bf16.py main()
```python
def main(fp8_path, bf16_path)
```

**Quick Example:**
```bash
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8 --output-bf16-hf-path /path/to/bf16
```

---

## 🎛️ Configuration and Setup Functions

### Global Configuration

| Variable | Module | Purpose | Default |
|----------|--------|---------|---------|
| `world_size` | `model.py` | Number of distributed processes | 1 |
| `rank` | `model.py` | Current process rank | 0 |
| `block_size` | `model.py` | Quantization block size | 128 |
| `gemm_impl` | `model.py` | GEMM implementation | "bf16" |
| `attn_impl` | `model.py` | Attention implementation | "absorb" |

### Setup Functions

| Function | Module | Purpose | When to Use |
|----------|--------|---------|-------------|
| `torch.set_default_dtype()` | PyTorch | Set default tensor dtype | Before model initialization |
| `torch.set_default_device()` | PyTorch | Set default device | Before model initialization |
| `dist.init_process_group()` | PyTorch | Initialize distributed training | Multi-GPU setups |

---

## 🚀 Performance Functions

### Optimization Utilities

| Function | Purpose | Example Usage |
|----------|---------|---------------|
| `torch.compile()` | Model compilation for speed | `model = torch.compile(model, mode="reduce-overhead")` |
| `torch.inference_mode()` | Disable gradients for inference | `with torch.inference_mode(): ...` |
| `torch.autocast()` | Mixed precision computation | `with torch.autocast('cuda', dtype=torch.bfloat16): ...` |

### Memory Management

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `torch.cuda.empty_cache()` | Clear GPU memory cache | After large operations |
| `torch.cuda.synchronize()` | Synchronize GPU operations | Before timing measurements |
| `torch.cuda.memory_allocated()` | Check memory usage | Memory debugging |

---

## 🔧 Utility Functions

### Helper Functions

| Function | Module | Purpose | Signature |
|----------|--------|---------|-----------|
| `find_correction_dim()` | `model.py` | YARN scaling calculation | Internal helper function |
| `find_correction_range()` | `model.py` | YARN range calculation | Internal helper function |
| `linear_ramp_factor()` | `model.py` | Linear interpolation | Internal helper function |

### Validation Functions

| Function | Purpose | Quick Example |
|----------|---------|---------------|
| Tensor contiguity check | Ensure optimal memory layout | `assert tensor.is_contiguous()` |
| Dimension validation | Check tensor dimensions | `assert tensor.dim() == 2` |
| Device consistency | Verify tensor devices | `assert tensor.is_cuda` |

---

## 📊 Monitoring and Debugging Functions

### Performance Monitoring

```python
# Example monitoring functions (from usage examples)
def benchmark_generation(model, tokenizer, prompts, iterations=10)
def measure_memory_usage()
def profile_generation_performance()
```

### Debugging Utilities

```python
# Example debugging functions (from usage examples)
def debug_model_state(model)
def debug_generation(model, tokenizer, prompt, debug=True)
def validate_conversion(original_path, converted_path, tolerance=1e-3)
```

---

## 🔗 Function Dependencies

### Dependency Graph

```
Transformer
├── Block
│   ├── MLA
│   │   ├── Linear layers (ColumnParallel, RowParallel)
│   │   ├── RMSNorm
│   │   └── apply_rotary_emb()
│   └── MoE / MLP
│       ├── Gate (for MoE)
│       ├── Expert (for MoE)
│       └── Linear layers
├── ParallelEmbedding
├── RMSNorm
└── precompute_freqs_cis()

generate()
├── Transformer.forward()
├── sample()
└── Tokenizer operations

Quantization Pipeline
├── act_quant() → act_quant_kernel()
├── weight_dequant() → weight_dequant_kernel()
└── fp8_gemm() → fp8_gemm_kernel()

Conversion Pipeline
├── convert.py → Parameter mapping + Expert distribution
└── fp8_cast_bf16.py → weight_dequant() + File management
```

---

## 📋 Function Checklists

### Before Using Generation Functions

- [ ] Model is properly initialized
- [ ] Weights are loaded
- [ ] Tokenizer is available
- [ ] Input tokens are properly formatted
- [ ] GPU memory is sufficient

### Before Using Quantization Functions

- [ ] Tensors are contiguous
- [ ] Tensors are on CUDA device
- [ ] Dimensions are compatible with block size
- [ ] Scaling factors are available (for dequantization)

### Before Using Conversion Functions

- [ ] Input model exists and is accessible
- [ ] Sufficient disk space available
- [ ] Model parallelism settings are valid
- [ ] Output directory is writable
- [ ] Dependencies are installed

---

## 🎯 Common Function Combinations

### Text Generation Workflow

```python
# 1. Load configuration
args = ModelArgs(**config)

# 2. Initialize model
model = Transformer(args)

# 3. Load weights
load_model(model, checkpoint_path)

# 4. Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# 5. Generate text
prompt_tokens = [tokenizer.encode(prompt)]
completions = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)

# 6. Decode results
text = tokenizer.decode(completions[0], skip_special_tokens=True)
```

### Quantization Workflow

```python
# 1. Prepare tensors
x = x.contiguous()
assert x.size(-1) % block_size == 0

# 2. Quantize activations
x_quant, x_scales = act_quant(x, block_size)

# 3. Perform FP8 computation
result = fp8_gemm(x_quant, x_scales, weight_quant, weight_scales)

# 4. Use result in further computations
```

### Model Conversion Workflow

```python
# 1. Convert from Hugging Face format
convert_main(hf_path, converted_path, n_experts, mp)

# 2. Optional: Convert precision
if target_precision == "bf16":
    fp8_to_bf16_main(converted_path, bf16_path)

# 3. Validate conversion
validate_conversion(original_path, final_path)
```

---

## 🎪 Advanced Function Patterns

### Context Managers

```python
# Inference optimization context
@contextmanager
def optimized_inference():
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            yield

# Usage
with optimized_inference():
    result = model(tokens)
```

### Decorator Patterns

```python
# Performance monitoring decorator
def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.3f}s")
        return result
    return wrapper

# Usage
@monitor_performance
def my_generation_function():
    return generate(model, prompt_tokens, 100, eos_id)
```

### Error Handling Patterns

```python
# Robust function calling with retry
def robust_generate(model, prompt_tokens, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return generate(model, prompt_tokens, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == max_retries - 1:
                raise
            print(f"OOM error, retrying {attempt + 1}/{max_retries}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Error on attempt {attempt + 1}: {e}")
```

---

## 📚 Function Categories by Use Case

### 🎮 Interactive Usage

| Function | Purpose | Quick Start |
|----------|---------|-------------|
| `main()` (generate.py) | Interactive chat | `python generate.py --interactive` |
| `generate()` | Text completion | See [Basic Examples](./USAGE_EXAMPLES.md#basic-usage-examples) |
| `sample()` | Custom sampling | For advanced generation control |

### 🏭 Production Deployment

| Function | Purpose | Documentation |
|----------|---------|---------------|
| `Transformer.__init__()` | Model initialization | [API Docs](./API_DOCUMENTATION.md#transformer) |
| `generate()` with batching | Batch inference | [Usage Examples](./USAGE_EXAMPLES.md#batch-processing) |
| Distributed setup functions | Multi-GPU deployment | [Distributed Guide](./USAGE_EXAMPLES.md#distributed-computing) |

### 🔬 Research and Development

| Function | Purpose | Documentation |
|----------|---------|---------------|
| `ModelArgs` customization | Custom architectures | [API Docs](./API_DOCUMENTATION.md#modelargs) |
| Quantization functions | Performance research | [Kernel Docs](./KERNEL_DOCUMENTATION.md) |
| Component analysis | Architecture studies | [API Docs](./API_DOCUMENTATION.md#model-architecture) |

### 🛠️ DevOps and Infrastructure

| Function | Purpose | Documentation |
|----------|---------|---------------|
| `convert.py main()` | Model format conversion | [Conversion Docs](./CONVERSION_DOCUMENTATION.md) |
| `fp8_cast_bf16.py main()` | Precision conversion | [Conversion Docs](./CONVERSION_DOCUMENTATION.md) |
| Validation functions | Quality assurance | [Conversion Docs](./CONVERSION_DOCUMENTATION.md#validation-and-testing) |

---

## 🎯 Function Selection Guide

### Choose the Right Function for Your Task

#### Text Generation Tasks

| Task | Recommended Function | Example |
|------|---------------------|---------|
| Simple completion | `generate()` | Single prompt completion |
| Interactive chat | `main()` with `--interactive` | Chat interface |
| Batch processing | `generate()` with multiple prompts | Bulk text generation |
| Streaming output | Custom implementation | Real-time token streaming |

#### Model Optimization Tasks

| Task | Recommended Function | Example |
|------|---------------------|---------|
| Memory reduction | `act_quant()` | Quantize activations |
| Speed optimization | `fp8_gemm()` | Fast matrix operations |
| Custom precision | `weight_dequant()` | Convert quantized weights |

#### Deployment Tasks

| Task | Recommended Function | Example |
|------|---------------------|---------|
| Format conversion | `convert.py main()` | HF to DeepSeek format |
| Precision conversion | `fp8_cast_bf16.py main()` | FP8 to BF16 |
| Multi-GPU setup | Distributed functions | Scale across GPUs |

---

## 🔍 Function Search Index

### By Input Type

#### Torch.Tensor Input
- `act_quant()` - Quantize tensor
- `weight_dequant()` - Dequantize tensor  
- `apply_rotary_emb()` - Apply positional encoding
- `sample()` - Sample from logits
- All model forward methods

#### String/Path Input
- `main()` (generate.py) - File paths for models/configs
- `convert.py main()` - Model directory paths
- `fp8_cast_bf16.py main()` - Model directory paths

#### List Input
- `generate()` - List of token sequences
- Batch processing functions

### By Output Type

#### Torch.Tensor Output
- Most model components (`forward()` methods)
- Quantization functions
- Matrix operations

#### List[List[int]] Output
- `generate()` - Generated token sequences

#### None Output
- Setup and conversion functions
- Configuration functions

### By Computational Complexity

#### O(1) - Constant Time
- Configuration functions
- Simple tensor operations
- Utility functions

#### O(n) - Linear Time
- Token sampling
- Simple tensor transformations
- File operations

#### O(n²) - Quadratic Time
- Attention computations
- Matrix multiplications
- Large model operations

#### O(n³) - Cubic Time
- Batch matrix operations
- Complex model computations

---

## 🎯 Performance Function Matrix

| Function | CPU Usage | GPU Usage | Memory Usage | Typical Runtime |
|----------|-----------|-----------|--------------|-----------------|
| `ModelArgs.__init__()` | Low | None | Minimal | < 1ms |
| `Transformer.__init__()` | Medium | Low | High | 1-10s |
| `generate()` | Low | High | Medium | 1-60s |
| `act_quant()` | Low | High | Medium | 1-100ms |
| `fp8_gemm()` | Low | High | Medium | 1-50ms |
| `convert.py main()` | High | Low | High | 5-60min |
| `fp8_cast_bf16.py main()` | Medium | High | High | 10-120min |

---

## 📖 Learning Path by Function Complexity

### Beginner Level
1. `ModelArgs` - Understanding configuration
2. `generate()` - Basic text generation
3. `sample()` - Token sampling
4. `main()` (generate.py) - Command-line usage

### Intermediate Level
1. `Transformer` - Model architecture
2. `Block` and `MLA` - Attention mechanisms
3. `MoE` - Mixture of experts
4. Linear layer variants - Distributed computing

### Advanced Level
1. Quantization functions - Performance optimization
2. Triton kernels - Low-level GPU programming
3. Conversion utilities - Model deployment
4. Custom implementations - Research and development

---

## 🎯 Quick Reference Cards

### Essential Functions Card

```python
# Model Setup
args = ModelArgs(**config)
model = Transformer(args)

# Text Generation  
completions = generate(model, prompt_tokens, max_new_tokens, eos_id, temperature)

# Quantization
x_quant, scales = act_quant(x, block_size)
result = fp8_gemm(a_quant, a_scales, b_quant, b_scales)

# Conversion
convert_main(hf_path, output_path, n_experts, mp)
```

### Error Handling Card

```python
# Common error checks
assert tensor.is_contiguous()
assert tensor.is_cuda
assert tensor.size(-1) % block_size == 0
assert world_size > 0

# Memory management
torch.cuda.empty_cache()
if torch.cuda.memory_allocated() > threshold:
    # Take action
```

### Performance Card

```python
# Speed optimizations
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.matmul.allow_tf32 = True
model = torch.compile(model)

# Memory optimizations  
with torch.inference_mode():
    with torch.autocast('cuda'):
        # Inference code
```

---

*This function reference provides quick access to all public APIs in the DeepSeek-V3 codebase. For detailed documentation of each function, refer to the specific documentation files linked throughout this guide.*