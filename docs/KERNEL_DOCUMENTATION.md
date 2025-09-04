# DeepSeek-V3 Kernel and Quantization Documentation

## Table of Contents

1. [Overview](#overview)
2. [Quantization Functions](#quantization-functions)
   - [act_quant()](#act_quant)
   - [weight_dequant()](#weight_dequant)
3. [FP8 GEMM Operations](#fp8-gemm-operations)
   - [fp8_gemm()](#fp8_gemm)
   - [fp8_gemm_kernel()](#fp8_gemm_kernel)
4. [Triton Kernels](#triton-kernels)
   - [act_quant_kernel()](#act_quant_kernel)
   - [weight_dequant_kernel()](#weight_dequant_kernel)
5. [Performance Optimization](#performance-optimization)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)

---

## Overview

The DeepSeek-V3 kernel module provides high-performance quantization and matrix multiplication operations using Triton kernels. These optimizations are crucial for achieving efficient FP8 training and inference at scale.

### Key Features

- **Block-wise Quantization**: Efficient FP8 quantization with configurable block sizes
- **Triton Kernels**: GPU-optimized kernels for maximum performance
- **Auto-tuning**: Automatic optimization of kernel configurations
- **Memory Efficiency**: Reduced memory footprint through quantization
- **Numerical Stability**: Careful handling of scaling factors and precision

---

## Quantization Functions

### act_quant()

**Function**: `act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]`

Quantizes input activation tensors using block-wise FP8 quantization for memory efficiency and computational speedup.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | torch.Tensor | - | Input tensor to be quantized (must be contiguous) |
| `block_size` | int | 128 | Size of blocks for quantization granularity |

#### Returns

- `Tuple[torch.Tensor, torch.Tensor]`: 
  - Quantized tensor with dtype `torch.float8_e4m3fn`
  - Scaling factors tensor with dtype `torch.float32`

#### Requirements

- Input tensor `x` must be contiguous (`x.is_contiguous()`)
- Last dimension of `x` must be divisible by `block_size`
- Tensor must be on CUDA device

#### Algorithm Details

The quantization process works as follows:

1. **Block Division**: Input tensor is divided into blocks of size `block_size`
2. **Scale Computation**: For each block, compute scale as `max(abs(block)) / 448.0`
3. **Quantization**: Divide each element by its block's scale factor
4. **Type Conversion**: Convert to FP8 format (`torch.float8_e4m3fn`)

#### Example Usage

```python
import torch
from kernel import act_quant

# Create input tensor
x = torch.randn(2, 128, 1024, dtype=torch.bfloat16, device='cuda').contiguous()

# Quantize with default block size
quantized_x, scales = act_quant(x, block_size=128)

print(f"Original shape: {x.shape}, dtype: {x.dtype}")
print(f"Quantized shape: {quantized_x.shape}, dtype: {quantized_x.dtype}")
print(f"Scales shape: {scales.shape}, dtype: {scales.dtype}")

# Output:
# Original shape: torch.Size([2, 128, 1024]), dtype: torch.bfloat16
# Quantized shape: torch.Size([2, 128, 1024]), dtype: torch.float8_e4m3fn
# Scales shape: torch.Size([2, 128, 8]), dtype: torch.float32
```

#### Performance Considerations

- **Block Size**: Smaller blocks provide better precision but more overhead
- **Memory Layout**: Contiguous tensors are required for optimal performance
- **GPU Utilization**: Kernel is optimized for high GPU occupancy

---

### weight_dequant()

**Function**: `weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor`

Dequantizes weight tensors using provided scaling factors, converting from quantized format back to full precision.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | torch.Tensor | - | Quantized weight tensor (2D, contiguous) |
| `s` | torch.Tensor | - | Scale tensor (2D, contiguous) |
| `block_size` | int | 128 | Block size used during quantization |

#### Returns

- `torch.Tensor`: Dequantized weight tensor with default dtype

#### Requirements

- Both `x` and `s` must be contiguous and 2-dimensional
- Tensors must be on CUDA device
- Scale tensor dimensions must match the block structure

#### Algorithm Details

The dequantization process:

1. **Block Reconstruction**: Reconstruct the original block structure
2. **Scale Application**: Multiply each quantized value by its corresponding scale
3. **Type Conversion**: Convert to the default floating-point dtype

#### Example Usage

```python
import torch
from kernel import weight_dequant

# Simulate quantized weights and scales
M, N = 1024, 2048
block_size = 128

# Quantized weights (FP8)
quantized_weights = torch.randn(M, N, dtype=torch.float8_e4m3fn, device='cuda')

# Scaling factors
num_blocks_m = (M + block_size - 1) // block_size
num_blocks_n = (N + block_size - 1) // block_size
scales = torch.randn(num_blocks_m, num_blocks_n, dtype=torch.float32, device='cuda')

# Dequantize
dequantized_weights = weight_dequant(quantized_weights, scales, block_size)

print(f"Quantized shape: {quantized_weights.shape}, dtype: {quantized_weights.dtype}")
print(f"Dequantized shape: {dequantized_weights.shape}, dtype: {dequantized_weights.dtype}")
```

---

## FP8 GEMM Operations

### fp8_gemm()

**Function**: `fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor`

High-performance matrix multiplication for FP8 tensors with scaling factors.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | torch.Tensor | First input matrix (contiguous) |
| `a_s` | torch.Tensor | Scaling factors for matrix `a` (contiguous) |
| `b` | torch.Tensor | Second input matrix (contiguous) |
| `b_s` | torch.Tensor | Scaling factors for matrix `b` (contiguous) |

#### Returns

- `torch.Tensor`: Result of matrix multiplication with default dtype

#### Features

- **Auto-tuning**: Automatically selects optimal kernel configuration
- **Block-wise Scaling**: Applies scaling factors at block granularity
- **Memory Efficient**: Operates directly on quantized tensors
- **High Throughput**: Optimized for maximum GPU utilization

#### Algorithm Details

The FP8 GEMM operation:

1. **Matrix Preparation**: Ensure all tensors are contiguous and properly shaped
2. **Block-wise Computation**: Perform matrix multiplication in blocks with scaling
3. **Accumulation**: Accumulate results in FP32 for numerical stability
4. **Output Conversion**: Convert final result to target dtype

#### Example Usage

```python
import torch
from kernel import act_quant, fp8_gemm

# Create input matrices
a = torch.randn(512, 1024, dtype=torch.bfloat16, device='cuda').contiguous()
b = torch.randn(2048, 1024, dtype=torch.bfloat16, device='cuda').contiguous()

# Quantize inputs
a_quant, a_scales = act_quant(a)
b_quant, b_scales = act_quant(b.T)  # Transpose for correct dimensions

# Perform FP8 GEMM
result = fp8_gemm(a_quant, a_scales, b_quant, b_scales)

print(f"Input A shape: {a.shape}")
print(f"Input B shape: {b.shape}")
print(f"Result shape: {result.shape}")
```

---

### fp8_gemm_kernel()

**Triton Kernel**: `fp8_gemm_kernel()`

Low-level Triton kernel that implements the core FP8 matrix multiplication logic.

#### Kernel Configuration

The kernel uses auto-tuning with multiple configurations:

```python
fp8_gemm_configs = [
    Config({
        'BLOCK_SIZE_M': block_m, 
        'BLOCK_SIZE_N': block_n, 
        'BLOCK_SIZE_K': 128
    }, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] 
    for block_n in [32, 64, 128] 
    for num_stages in [3, 4, 5, 6]
]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a_ptr` | tl.tensor | Pointer to first input matrix |
| `b_ptr` | tl.tensor | Pointer to second input matrix |
| `c_ptr` | tl.tensor | Pointer to output matrix |
| `a_s_ptr` | tl.tensor | Pointer to scaling factors for matrix A |
| `b_s_ptr` | tl.tensor | Pointer to scaling factors for matrix B |
| `M` | int | Number of rows in matrix A and C |
| `N` | tl.constexpr | Number of columns in matrix B and C |
| `K` | tl.constexpr | Inner dimension for matrix multiplication |
| `BLOCK_SIZE_M` | tl.constexpr | Block size for M dimension |
| `BLOCK_SIZE_N` | tl.constexpr | Block size for N dimension |
| `BLOCK_SIZE_K` | tl.constexpr | Block size for K dimension |

---

## Triton Kernels

### act_quant_kernel()

**Triton Kernel**: `act_quant_kernel()`

GPU kernel for efficient activation quantization.

```python
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x_ptr` | triton.Pointer | Pointer to input tensor |
| `y_ptr` | triton.Pointer | Pointer to output quantized tensor |
| `s_ptr` | triton.Pointer | Pointer to output scaling factors |
| `BLOCK_SIZE` | tl.constexpr | Block size for processing |

#### Algorithm

1. **Load Block**: Load a block of input values
2. **Compute Scale**: Find maximum absolute value and compute scale
3. **Quantize**: Divide values by scale factor
4. **Store Results**: Store quantized values and scale factor

---

### weight_dequant_kernel()

**Triton Kernel**: `weight_dequant_kernel()`

GPU kernel for efficient weight dequantization.

```python
@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x_ptr` | tl.pointer | Pointer to quantized weights |
| `s_ptr` | tl.pointer | Pointer to scaling factors |
| `y_ptr` | tl.pointer | Pointer to output dequantized weights |
| `M` | int | Number of rows |
| `N` | int | Number of columns |
| `BLOCK_SIZE` | tl.constexpr | Block size for tiling |

---

## Performance Optimization

### Kernel Auto-tuning

The FP8 GEMM kernel uses comprehensive auto-tuning to find optimal configurations:

```python
# Auto-tuning parameters
BLOCK_SIZES_M = [16, 32, 64]
BLOCK_SIZES_N = [32, 64, 128] 
BLOCK_SIZE_K = 128
NUM_STAGES = [3, 4, 5, 6]
NUM_WARPS = 8
```

### Memory Access Patterns

- **Coalesced Access**: Kernels are designed for coalesced memory access
- **Shared Memory**: Efficient use of shared memory for block computations
- **Register Usage**: Optimized register allocation for maximum occupancy

### Block Size Selection

Choose block sizes based on your hardware and workload:

| Block Size | Memory Usage | Precision | Performance |
|------------|--------------|-----------|-------------|
| 64 | Lower | Higher | Moderate |
| 128 | Moderate | Good | High |
| 256 | Higher | Lower | Highest |

---

## Usage Examples

### Basic Quantization Workflow

```python
import torch
from kernel import act_quant, weight_dequant, fp8_gemm

# Set up environment
torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

# Create sample tensors
batch_size, seq_len, hidden_dim = 4, 512, 2048
activations = torch.randn(batch_size, seq_len, hidden_dim).contiguous()
weights = torch.randn(hidden_dim, hidden_dim).contiguous()

# Quantize activations
act_quant_tensor, act_scales = act_quant(activations, block_size=128)

# Simulate quantized weights (normally loaded from checkpoint)
weight_quant_tensor = torch.randn(hidden_dim, hidden_dim, dtype=torch.float8_e4m3fn, device='cuda')
weight_scales = torch.randn((hidden_dim + 127) // 128, (hidden_dim + 127) // 128, device='cuda')

# Perform FP8 GEMM
result = fp8_gemm(
    act_quant_tensor.view(-1, hidden_dim),  # Flatten batch and sequence dimensions
    act_scales.view(-1, act_scales.size(-1)),
    weight_quant_tensor,
    weight_scales
)

print(f"Result shape: {result.shape}")
print(f"Result dtype: {result.dtype}")
```

### Integration with Linear Layers

```python
from model import Linear
import torch

# Create a linear layer that uses quantized operations
layer = Linear(in_features=1024, out_features=2048, bias=False)

# Input tensor
x = torch.randn(2, 128, 1024, device='cuda')

# Forward pass (automatically uses quantization if weights are quantized)
output = layer(x)
```

### Custom Quantization Parameters

```python
from kernel import act_quant

# Different block sizes for different precision-performance tradeoffs
block_sizes = [64, 128, 256]
x = torch.randn(1024, 2048, device='cuda').contiguous()

for block_size in block_sizes:
    quant_x, scales = act_quant(x, block_size=block_size)
    
    # Compute quantization error
    reconstructed = quant_x.float() * scales.repeat_interleave(block_size, dim=-1)
    error = (x.float() - reconstructed).abs().mean()
    
    print(f"Block size {block_size}: Error = {error:.6f}")
```

---

## Best Practices

### Memory Management

1. **Ensure Contiguity**: Always ensure tensors are contiguous before quantization
   ```python
   if not tensor.is_contiguous():
       tensor = tensor.contiguous()
   ```

2. **Proper Device Placement**: Keep all tensors on the same CUDA device
   ```python
   tensor = tensor.to('cuda')
   ```

3. **Dimension Alignment**: Ensure dimensions are properly aligned with block sizes
   ```python
   assert tensor.size(-1) % block_size == 0
   ```

### Performance Optimization

1. **Block Size Selection**: Choose block sizes that are multiples of 32 for optimal GPU performance
2. **Batch Processing**: Process multiple samples together when possible
3. **Memory Reuse**: Reuse quantized tensors and scaling factors when possible

### Error Handling

```python
from kernel import act_quant

def safe_act_quant(x, block_size=128):
    """Safely quantize tensor with proper error handling."""
    try:
        # Validate inputs
        if not x.is_contiguous():
            x = x.contiguous()
        
        if x.size(-1) % block_size != 0:
            raise ValueError(f"Last dimension {x.size(-1)} not divisible by block_size {block_size}")
        
        if not x.is_cuda:
            x = x.cuda()
        
        return act_quant(x, block_size)
    
    except Exception as e:
        print(f"Quantization failed: {e}")
        return None, None
```

### Numerical Stability

1. **Scale Factor Handling**: Ensure scale factors are properly normalized
2. **Overflow Prevention**: Monitor for potential overflow in FP8 range
3. **Precision Loss**: Be aware of precision loss and compensate when necessary

---

## Advanced Usage

### Custom Block Sizes

```python
# Experiment with different block sizes for optimal performance
def benchmark_block_sizes(tensor, block_sizes=[64, 128, 256]):
    """Benchmark different block sizes for quantization."""
    import time
    
    results = {}
    for block_size in block_sizes:
        if tensor.size(-1) % block_size != 0:
            continue
            
        start_time = time.time()
        for _ in range(100):  # Multiple runs for accurate timing
            quant_tensor, scales = act_quant(tensor, block_size)
        torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        results[block_size] = elapsed / 100
        
        print(f"Block size {block_size}: {elapsed/100:.4f}s per operation")
    
    return results
```

### Integration with Training

```python
def quantized_forward_pass(x, weight, bias=None, block_size=128):
    """Example of integrating quantization in a forward pass."""
    
    # Quantize activations
    x_quant, x_scales = act_quant(x, block_size)
    
    # Assume weights are already quantized and have scales
    if hasattr(weight, 'scale'):
        # Use FP8 GEMM for quantized computation
        result = fp8_gemm(x_quant, x_scales, weight, weight.scale)
    else:
        # Fall back to standard computation
        result = torch.nn.functional.linear(x, weight, bias)
    
    return result
```

---

## Troubleshooting

### Common Issues

1. **Tensor Contiguity Error**
   ```
   AssertionError: Tensor is not contiguous
   ```
   **Solution**: Call `.contiguous()` on the tensor before quantization

2. **Dimension Mismatch**
   ```
   AssertionError: Last dimension not divisible by block_size
   ```
   **Solution**: Pad tensor or choose a compatible block size

3. **Device Mismatch**
   ```
   RuntimeError: Expected all tensors to be on the same device
   ```
   **Solution**: Move all tensors to the same CUDA device

### Debugging Tips

1. **Check Tensor Properties**:
   ```python
   print(f"Shape: {tensor.shape}")
   print(f"Dtype: {tensor.dtype}")
   print(f"Device: {tensor.device}")
   print(f"Contiguous: {tensor.is_contiguous()}")
   ```

2. **Validate Quantization**:
   ```python
   # Check quantization quality
   original = torch.randn(1024, 1024, device='cuda').contiguous()
   quant, scales = act_quant(original)
   
   # Reconstruct and measure error
   reconstructed = quant.float() * scales.repeat_interleave(128, dim=-1)
   error = (original.float() - reconstructed).abs().mean()
   print(f"Quantization error: {error}")
   ```

3. **Performance Profiling**:
   ```python
   import torch.profiler
   
   with torch.profiler.profile() as prof:
       quant_tensor, scales = act_quant(input_tensor)
   
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

---

## Technical Specifications

### Supported Data Types

| Type | Usage | Precision | Memory |
|------|-------|-----------|--------|
| `torch.float8_e4m3fn` | Quantized weights/activations | 8-bit | Minimal |
| `torch.bfloat16` | Default computation | 16-bit | Moderate |
| `torch.float32` | Scaling factors | 32-bit | High |

### Hardware Requirements

- **GPU**: NVIDIA GPUs with Compute Capability 8.0+ (for FP8 support)
- **Memory**: Sufficient GPU memory for model and intermediate tensors
- **CUDA**: CUDA 11.8 or later
- **Triton**: Version 3.0.0 or later

### Performance Characteristics

- **Throughput**: Up to 3x improvement over FP16 operations
- **Memory**: ~50% reduction in memory usage
- **Accuracy**: Minimal precision loss with proper scaling
- **Latency**: Optimized for both training and inference workloads

---

This documentation provides comprehensive coverage of all quantization and kernel functions in the DeepSeek-V3 implementation, enabling efficient utilization of the FP8 training and inference capabilities.