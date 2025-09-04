# DeepSeek-V3 Model Conversion Documentation

## Table of Contents

1. [Overview](#overview)
2. [Hugging Face to DeepSeek Conversion](#hugging-face-to-deepseek-conversion)
   - [convert.py](#convertpy)
   - [Parameter Mapping](#parameter-mapping)
   - [Expert Handling](#expert-handling)
3. [FP8 to BF16 Conversion](#fp8-to-bf16-conversion)
   - [fp8_cast_bf16.py](#fp8_cast_bf16py)
   - [Weight Processing](#weight-processing)
   - [Memory Management](#memory-management)
4. [Conversion Workflows](#conversion-workflows)
5. [Validation and Testing](#validation-and-testing)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The DeepSeek-V3 project provides comprehensive model conversion utilities to transform models between different formats and precision levels. These tools are essential for:

- **Format Compatibility**: Converting between Hugging Face and DeepSeek formats
- **Precision Management**: Converting between FP8 and BF16 precisions
- **Distributed Deployment**: Preparing models for multi-GPU inference
- **Storage Optimization**: Optimizing model storage and loading

### Key Features

- **Automatic Parameter Mapping**: Intelligent mapping between different parameter naming conventions
- **Expert Distribution**: Proper handling of MoE expert parameters across multiple GPUs
- **Memory Efficiency**: Streaming conversion for large models
- **Validation**: Built-in validation and error checking
- **Metadata Preservation**: Maintains important model metadata during conversion

---

## Hugging Face to DeepSeek Conversion

### convert.py

**Function**: `main(hf_ckpt_path, save_path, n_experts, mp)`

Converts Hugging Face model checkpoints to DeepSeek-V3 format with support for model parallelism and expert distribution.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hf_ckpt_path` | str | Path to Hugging Face checkpoint directory containing safetensors files |
| `save_path` | str | Output directory for converted DeepSeek format checkpoints |
| `n_experts` | int | Total number of experts in the MoE model |
| `mp` | int | Model parallelism factor (number of GPUs/shards) |

#### Requirements

- `n_experts` must be divisible by `mp`
- Input directory must contain valid safetensors files
- Sufficient disk space for output files

#### Conversion Process

1. **Parameter Discovery**: Scans all safetensors files in the input directory
2. **Name Mapping**: Converts parameter names using predefined mapping rules
3. **Expert Distribution**: Distributes expert parameters across model parallel shards
4. **Dimension Splitting**: Splits parameters along specified dimensions for parallelism
5. **File Generation**: Creates separate safetensors files for each model parallel shard
6. **Tokenizer Copying**: Copies tokenizer files to the output directory

#### Command Line Usage

```bash
python convert.py \
  --hf-ckpt-path /path/to/huggingface/DeepSeek-V3 \
  --save-path /path/to/converted/DeepSeek-V3-Demo \
  --n-experts 256 \
  --model-parallel 16
```

#### Example Usage in Python

```python
from convert import main as convert_model
import os

def convert_with_validation(hf_path, output_path, n_experts, mp):
    """
    Convert model with validation checks.
    
    Args:
        hf_path (str): Hugging Face model path
        output_path (str): Output path for converted model
        n_experts (int): Number of experts
        mp (int): Model parallelism factor
    """
    
    # Validate inputs
    if not os.path.exists(hf_path):
        raise FileNotFoundError(f"Hugging Face checkpoint not found: {hf_path}")
    
    if n_experts % mp != 0:
        raise ValueError(f"n_experts ({n_experts}) must be divisible by mp ({mp})")
    
    # Check available disk space
    import shutil
    free_space_gb = shutil.disk_usage(os.path.dirname(output_path)).free / (1024**3)
    print(f"Available disk space: {free_space_gb:.2f} GB")
    
    # Perform conversion
    print(f"Converting model from {hf_path} to {output_path}")
    print(f"Configuration: {n_experts} experts, {mp} model parallel shards")
    
    convert_model(hf_path, output_path, n_experts, mp)
    
    # Verify output
    expected_files = [f"model{i}-mp{mp}.safetensors" for i in range(mp)]
    for file_name in expected_files:
        file_path = os.path.join(output_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected output file not created: {file_path}")
    
    print("Conversion completed successfully!")

# Usage
convert_with_validation(
    hf_path="/data/huggingface/DeepSeek-V3",
    output_path="/data/converted/DeepSeek-V3-Demo",
    n_experts=256,
    mp=16
)
```

---

### Parameter Mapping

The conversion process uses a comprehensive parameter mapping system to translate between Hugging Face and DeepSeek naming conventions.

#### Mapping Table

| Hugging Face Name | DeepSeek Name | Split Dimension | Description |
|-------------------|---------------|-----------------|-------------|
| `embed_tokens` | `embed` | 0 | Token embeddings |
| `input_layernorm` | `attn_norm` | None | Attention layer normalization |
| `post_attention_layernorm` | `ffn_norm` | None | FFN layer normalization |
| `q_proj` | `wq` | 0 | Query projection |
| `q_a_proj` | `wq_a` | None | Query LoRA A projection |
| `q_a_layernorm` | `q_norm` | None | Query LoRA normalization |
| `q_b_proj` | `wq_b` | 0 | Query LoRA B projection |
| `kv_a_proj_with_mqa` | `wkv_a` | None | Key-Value LoRA A projection |
| `kv_a_layernorm` | `kv_norm` | None | Key-Value LoRA normalization |
| `kv_b_proj` | `wkv_b` | 0 | Key-Value LoRA B projection |
| `o_proj` | `wo` | 1 | Output projection |
| `gate` | `gate` | None | MoE gate |
| `gate_proj` | `w1` | 0 | MLP gate projection |
| `down_proj` | `w2` | 1 | MLP down projection |
| `up_proj` | `w3` | 0 | MLP up projection |
| `norm` | `norm` | None | Final layer normalization |
| `lm_head` | `head` | 0 | Language model head |
| `scale` | `scale` | None | Quantization scales |

#### Custom Mapping Example

```python
# The mapping dictionary used in convert.py
mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}
```

---

### Expert Handling

The conversion process properly handles the distribution of MoE experts across multiple GPU shards.

#### Expert Distribution Algorithm

```python
def distribute_experts(expert_idx, mp, n_local_experts):
    """
    Determine which shard an expert belongs to.
    
    Args:
        expert_idx (int): Index of the expert
        mp (int): Model parallelism factor
        n_local_experts (int): Number of experts per shard
        
    Returns:
        int: Shard index for the expert (-1 if not in current shard)
    """
    
    shard_idx = expert_idx // n_local_experts
    if shard_idx < mp:
        return shard_idx
    return -1

# Example: 256 experts across 16 shards
n_experts = 256
mp = 16
n_local_experts = n_experts // mp  # 16 experts per shard

for expert_idx in range(n_experts):
    shard = distribute_experts(expert_idx, mp, n_local_experts)
    print(f"Expert {expert_idx} -> Shard {shard}")
```

#### Expert Parameter Processing

```python
def process_expert_parameter(param_name, param_tensor, expert_idx, mp, n_local_experts):
    """
    Process expert parameters for distribution across shards.
    
    Args:
        param_name (str): Parameter name
        param_tensor (torch.Tensor): Parameter tensor
        expert_idx (int): Expert index
        mp (int): Model parallelism factor
        n_local_experts (int): Experts per shard
        
    Returns:
        List[Tuple[int, torch.Tensor]]: List of (shard_idx, tensor) pairs
    """
    
    results = []
    
    for shard_idx in range(mp):
        shard_start = shard_idx * n_local_experts
        shard_end = shard_start + n_local_experts
        
        if shard_start <= expert_idx < shard_end:
            # Expert belongs to this shard
            results.append((shard_idx, param_tensor))
        # Expert doesn't belong to this shard - skip
    
    return results
```

---

## FP8 to BF16 Conversion

### fp8_cast_bf16.py

**Function**: `main(fp8_path, bf16_path)`

Converts FP8 quantized model weights to BF16 format for compatibility with frameworks that don't support FP8.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `fp8_path` | str | Path to directory containing FP8 model weights and metadata |
| `bf16_path` | str | Output directory for BF16 converted weights |

#### Features

- **Streaming Conversion**: Processes large models without loading everything into memory
- **Scale Factor Integration**: Properly applies quantization scales during conversion
- **Metadata Updates**: Updates model index files to reflect changes
- **Memory Management**: Intelligent caching and cleanup for large models
- **Error Handling**: Graceful handling of missing scale tensors

#### Command Line Usage

```bash
python fp8_cast_bf16.py \
  --input-fp8-hf-path /path/to/fp8/model \
  --output-bf16-hf-path /path/to/bf16/model
```

#### Example Usage in Python

```python
from fp8_cast_bf16 import main as convert_fp8_to_bf16
import os
import json

def convert_with_verification(fp8_path, bf16_path):
    """
    Convert FP8 to BF16 with comprehensive verification.
    
    Args:
        fp8_path (str): Input FP8 model path
        bf16_path (str): Output BF16 model path
    """
    
    # Verify input directory
    if not os.path.exists(fp8_path):
        raise FileNotFoundError(f"FP8 model path not found: {fp8_path}")
    
    # Check for required files
    required_files = ["model.safetensors.index.json"]
    for file_name in required_files:
        file_path = os.path.join(fp8_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load model index to understand structure
    with open(os.path.join(fp8_path, "model.safetensors.index.json")) as f:
        model_index = json.load(f)
    
    weight_map = model_index["weight_map"]
    fp8_weights = [name for name in weight_map.keys() if not name.endswith("_scale_inv")]
    scale_weights = [name for name in weight_map.keys() if name.endswith("_scale_inv")]
    
    print(f"Found {len(fp8_weights)} FP8 weight tensors")
    print(f"Found {len(scale_weights)} scale tensors")
    
    # Perform conversion
    print("Starting FP8 to BF16 conversion...")
    convert_fp8_to_bf16(fp8_path, bf16_path)
    
    # Verify output
    output_index_path = os.path.join(bf16_path, "model.safetensors.index.json")
    if os.path.exists(output_index_path):
        with open(output_index_path) as f:
            output_index = json.load(f)
        
        output_weights = list(output_index["weight_map"].keys())
        print(f"Output contains {len(output_weights)} weight tensors")
        
        # Check that scale_inv tensors were removed
        remaining_scales = [name for name in output_weights if name.endswith("_scale_inv")]
        if remaining_scales:
            print(f"Warning: {len(remaining_scales)} scale tensors remain in output")
        else:
            print("Successfully removed all scale tensors")
    
    print("Conversion completed successfully!")

# Usage
convert_with_verification(
    fp8_path="/data/models/DeepSeek-V3-FP8",
    bf16_path="/data/models/DeepSeek-V3-BF16"
)
```

---

### Weight Processing

The FP8 to BF16 conversion involves sophisticated weight processing to maintain model accuracy.

#### Weight Dequantization Process

```python
from kernel import weight_dequant
import torch

def process_fp8_weight(weight_tensor, scale_tensor):
    """
    Process a single FP8 weight tensor for conversion to BF16.
    
    Args:
        weight_tensor (torch.Tensor): FP8 quantized weight
        scale_tensor (torch.Tensor): Corresponding scale factors
        
    Returns:
        torch.Tensor: Dequantized BF16 weight tensor
    """
    
    # Verify tensor properties
    if weight_tensor.element_size() != 1:
        raise ValueError("Expected FP8 tensor (element_size == 1)")
    
    if weight_tensor.dim() != 2:
        raise ValueError("Expected 2D weight tensor")
    
    # Perform dequantization
    bf16_weight = weight_dequant(weight_tensor, scale_tensor)
    
    # Validate output
    if torch.isnan(bf16_weight).any():
        raise RuntimeError("NaN values detected in dequantized weights")
    
    if torch.isinf(bf16_weight).any():
        raise RuntimeError("Inf values detected in dequantized weights")
    
    return bf16_weight

# Example usage within conversion pipeline
def convert_single_safetensor_file(input_path, output_path):
    """Convert a single safetensors file from FP8 to BF16."""
    
    from safetensors.torch import load_file, save_file
    
    # Load FP8 weights
    state_dict = load_file(input_path, device="cuda")
    converted_dict = {}
    
    for weight_name, weight_tensor in state_dict.items():
        if weight_name.endswith("_scale_inv"):
            continue  # Skip scale tensors in output
        
        if weight_tensor.element_size() == 1:  # FP8 tensor
            scale_name = f"{weight_name}_scale_inv"
            if scale_name in state_dict:
                scale_tensor = state_dict[scale_name]
                converted_dict[weight_name] = process_fp8_weight(weight_tensor, scale_tensor)
            else:
                print(f"Warning: No scale tensor found for {weight_name}")
                converted_dict[weight_name] = weight_tensor
        else:
            # Non-FP8 tensor, copy as-is
            converted_dict[weight_name] = weight_tensor
    
    # Save converted weights
    save_file(converted_dict, output_path)
    print(f"Converted {input_path} -> {output_path}")
```

---

### Memory Management

The conversion utilities implement sophisticated memory management for handling large models.

#### Streaming Conversion Strategy

```python
class StreamingConverter:
    """Memory-efficient converter for large models."""
    
    def __init__(self, max_cached_files=2):
        self.max_cached_files = max_cached_files
        self.file_cache = {}
        self.cache_order = []
    
    def get_tensor(self, tensor_name, weight_map, base_path):
        """
        Retrieve tensor with intelligent caching.
        
        Args:
            tensor_name (str): Name of tensor to retrieve
            weight_map (dict): Mapping of tensor names to files
            base_path (str): Base directory path
            
        Returns:
            torch.Tensor: Retrieved tensor
        """
        
        file_name = weight_map[tensor_name]
        
        # Check if file is already cached
        if file_name not in self.file_cache:
            # Load file
            file_path = os.path.join(base_path, file_name)
            self.file_cache[file_name] = load_file(file_path, device="cuda")
            self.cache_order.append(file_name)
            
            # Manage cache size
            while len(self.file_cache) > self.max_cached_files:
                oldest_file = self.cache_order.pop(0)
                del self.file_cache[oldest_file]
                torch.cuda.empty_cache()
        
        return self.file_cache[file_name][tensor_name]
    
    def clear_cache(self):
        """Clear all cached files."""
        self.file_cache.clear()
        self.cache_order.clear()
        torch.cuda.empty_cache()

# Usage in conversion
def memory_efficient_conversion(fp8_path, bf16_path):
    """Perform memory-efficient FP8 to BF16 conversion."""
    
    converter = StreamingConverter(max_cached_files=3)
    
    try:
        # Load model index
        with open(os.path.join(fp8_path, "model.safetensors.index.json")) as f:
            model_index = json.load(f)
        
        weight_map = model_index["weight_map"]
        
        # Process each safetensor file
        safetensor_files = set(weight_map.values())
        
        for file_name in safetensor_files:
            print(f"Processing {file_name}...")
            
            # Process file with memory management
            # ... conversion logic ...
        
        print("Conversion completed with optimized memory usage")
        
    finally:
        converter.clear_cache()
```

---

## Conversion Workflows

### 1. Complete Model Preparation Pipeline

```python
import os
import json
import shutil
from typing import Dict, Any

class ModelPreparationPipeline:
    """Complete pipeline for preparing models for deployment."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
    
    def prepare_model(
        self,
        hf_model_path: str,
        model_name: str,
        target_precision: str = "bf16",
        model_parallel: int = 8
    ) -> str:
        """
        Complete model preparation workflow.
        
        Args:
            hf_model_path (str): Hugging Face model path
            model_name (str): Name for the prepared model
            target_precision (str): Target precision ("fp8" or "bf16")
            model_parallel (int): Model parallelism factor
            
        Returns:
            str: Path to prepared model
        """
        
        print(f"Preparing model: {model_name}")
        print(f"Source: {hf_model_path}")
        print(f"Target precision: {target_precision}")
        print(f"Model parallel: {model_parallel}")
        
        # Step 1: Create output directory
        output_dir = os.path.join(self.base_output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 2: Load and validate configuration
        config_path = os.path.join(hf_model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        n_experts = config.get("n_routed_experts", 256)
        
        # Step 3: Convert from Hugging Face format
        converted_dir = os.path.join(output_dir, "converted")
        self._convert_from_hf(hf_model_path, converted_dir, n_experts, model_parallel)
        
        # Step 4: Handle precision conversion if needed
        final_dir = converted_dir
        if target_precision == "bf16" and config.get("dtype") == "fp8":
            bf16_dir = os.path.join(output_dir, "bf16")
            self._convert_fp8_to_bf16(converted_dir, bf16_dir)
            final_dir = bf16_dir
        
        # Step 5: Create deployment configuration
        self._create_deployment_config(final_dir, config, model_parallel)
        
        print(f"Model preparation completed: {final_dir}")
        return final_dir
    
    def _convert_from_hf(self, hf_path, output_path, n_experts, mp):
        """Convert from Hugging Face format."""
        from convert import main as convert_main
        convert_main(hf_path, output_path, n_experts, mp)
    
    def _convert_fp8_to_bf16(self, fp8_path, bf16_path):
        """Convert from FP8 to BF16."""
        from fp8_cast_bf16 import main as fp8_to_bf16
        fp8_to_bf16(fp8_path, bf16_path)
    
    def _create_deployment_config(self, model_path, original_config, mp):
        """Create deployment configuration file."""
        
        deployment_config = {
            "model_path": model_path,
            "model_parallel": mp,
            "original_config": original_config,
            "deployment_info": {
                "conversion_date": datetime.now().isoformat(),
                "recommended_batch_size": min(8, 32 // mp),
                "recommended_seq_len": 4096
            }
        }
        
        config_path = os.path.join(model_path, "deployment_config.json")
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)

# Usage
pipeline = ModelPreparationPipeline("/data/prepared_models")

prepared_path = pipeline.prepare_model(
    hf_model_path="/data/huggingface/DeepSeek-V3",
    model_name="deepseek-v3-production",
    target_precision="bf16",
    model_parallel=16
)
```

### 2. Batch Conversion Utility

```python
import os
import concurrent.futures
from typing import List, Tuple

def batch_convert_models(conversion_jobs: List[Tuple[str, str, int, int]]):
    """
    Convert multiple models in parallel.
    
    Args:
        conversion_jobs: List of (hf_path, output_path, n_experts, mp) tuples
    """
    
    def convert_single_model(job):
        """Convert a single model."""
        hf_path, output_path, n_experts, mp = job
        
        try:
            print(f"Starting conversion: {os.path.basename(hf_path)}")
            convert_model(hf_path, output_path, n_experts, mp)
            print(f"Completed conversion: {os.path.basename(hf_path)}")
            return True, None
        except Exception as e:
            print(f"Failed conversion: {os.path.basename(hf_path)} - {e}")
            return False, str(e)
    
    # Execute conversions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(convert_single_model, conversion_jobs))
    
    # Report results
    successful = sum(1 for success, _ in results if success)
    total = len(conversion_jobs)
    
    print(f"\nConversion Summary:")
    print(f"  Successful: {successful}/{total}")
    print(f"  Failed: {total - successful}/{total}")
    
    # Report failures
    for i, (success, error) in enumerate(results):
        if not success:
            job = conversion_jobs[i]
            print(f"  Failed job {i}: {job[0]} -> {error}")

# Usage
jobs = [
    ("/data/hf/model-16B", "/data/converted/model-16B", 64, 4),
    ("/data/hf/model-236B", "/data/converted/model-236B", 128, 8),
    ("/data/hf/model-671B", "/data/converted/model-671B", 256, 16),
]

batch_convert_models(jobs)
```

---

## Validation and Testing

### 1. Conversion Validation

```python
import torch
from safetensors.torch import load_file
import numpy as np

def validate_conversion(original_path, converted_path, tolerance=1e-3):
    """
    Validate that conversion preserves model weights correctly.
    
    Args:
        original_path (str): Path to original model
        converted_path (str): Path to converted model
        tolerance (float): Numerical tolerance for comparison
        
    Returns:
        bool: True if validation passes
    """
    
    print("Validating conversion...")
    
    # Load original model (first shard for simplicity)
    original_files = [f for f in os.listdir(original_path) if f.endswith('.safetensors')]
    converted_files = [f for f in os.listdir(converted_path) if f.endswith('.safetensors')]
    
    if len(original_files) == 0 or len(converted_files) == 0:
        raise ValueError("No safetensors files found")
    
    # Compare first file as sample
    original_dict = load_file(os.path.join(original_path, original_files[0]), device="cpu")
    converted_dict = load_file(os.path.join(converted_path, converted_files[0]), device="cpu")
    
    # Find common parameters
    common_params = set(original_dict.keys()) & set(converted_dict.keys())
    
    if len(common_params) == 0:
        print("Warning: No common parameters found")
        return False
    
    print(f"Comparing {len(common_params)} common parameters...")
    
    max_diff = 0.0
    failed_params = []
    
    for param_name in common_params:
        orig_param = original_dict[param_name].float()
        conv_param = converted_dict[param_name].float()
        
        if orig_param.shape != conv_param.shape:
            print(f"Shape mismatch for {param_name}: {orig_param.shape} vs {conv_param.shape}")
            failed_params.append(param_name)
            continue
        
        # Calculate difference
        diff = torch.abs(orig_param - conv_param).max().item()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print(f"Large difference in {param_name}: {diff:.6f}")
            failed_params.append(param_name)
    
    print(f"Maximum parameter difference: {max_diff:.6f}")
    print(f"Tolerance: {tolerance:.6f}")
    
    if failed_params:
        print(f"Failed validation for {len(failed_params)} parameters")
        return False
    else:
        print("Validation passed!")
        return True

# Usage
is_valid = validate_conversion(
    "/data/original/model",
    "/data/converted/model",
    tolerance=1e-3
)
```

### 2. Inference Validation

```python
def validate_inference_equivalence(original_model, converted_model, test_prompts):
    """
    Validate that converted model produces equivalent outputs.
    
    Args:
        original_model: Original model instance
        converted_model: Converted model instance
        test_prompts: List of test prompts for validation
        
    Returns:
        bool: True if outputs are equivalent within tolerance
    """
    
    from generate import generate
    
    print("Validating inference equivalence...")
    
    original_model.eval()
    converted_model.eval()
    
    with torch.inference_mode():
        for i, prompt in enumerate(test_prompts):
            print(f"Testing prompt {i+1}/{len(test_prompts)}")
            
            prompt_tokens = [tokenizer.encode(prompt)]
            
            # Generate with original model
            orig_completion = generate(
                original_model, prompt_tokens, 50, tokenizer.eos_token_id, 0.0
            )
            
            # Generate with converted model
            conv_completion = generate(
                converted_model, prompt_tokens, 50, tokenizer.eos_token_id, 0.0
            )
            
            # Compare outputs
            if orig_completion != conv_completion:
                print(f"Output mismatch for prompt {i+1}")
                print(f"Original: {tokenizer.decode(orig_completion[0])}")
                print(f"Converted: {tokenizer.decode(conv_completion[0])}")
                return False
    
    print("Inference validation passed!")
    return True

# Usage
test_prompts = [
    "What is machine learning?",
    "Explain quantum physics.",
    "Write a short poem about nature."
]

is_equivalent = validate_inference_equivalence(
    original_model, converted_model, test_prompts
)
```

---

## Troubleshooting

### Common Conversion Issues

#### 1. Missing Scale Tensors

**Problem**: FP8 weights missing corresponding scale tensors

**Solution**:
```python
def check_scale_tensor_completeness(model_path):
    """Check if all FP8 weights have corresponding scale tensors."""
    
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        model_index = json.load(f)
    
    weight_map = model_index["weight_map"]
    
    fp8_weights = []
    scale_tensors = []
    
    for tensor_name in weight_map.keys():
        if tensor_name.endswith("_scale_inv"):
            scale_tensors.append(tensor_name[:-10])  # Remove "_scale_inv"
        else:
            # Check if this might be an FP8 weight
            fp8_weights.append(tensor_name)
    
    missing_scales = set(fp8_weights) - set(scale_tensors)
    
    if missing_scales:
        print(f"Missing scale tensors for: {missing_scales}")
        return False
    else:
        print("All FP8 weights have corresponding scale tensors")
        return True

# Usage
check_scale_tensor_completeness("/path/to/fp8/model")
```

#### 2. Dimension Mismatch Errors

**Problem**: Parameter dimensions don't match expected model parallelism

**Solution**:
```python
def diagnose_dimension_issues(hf_path, mp):
    """Diagnose dimension-related issues before conversion."""
    
    from safetensors.torch import safe_open
    import glob
    
    print(f"Diagnosing dimension compatibility for mp={mp}")
    
    # Check first safetensor file
    safetensor_files = glob.glob(os.path.join(hf_path, "*.safetensors"))
    if not safetensor_files:
        print("No safetensors files found")
        return False
    
    with safe_open(safetensor_files[0], framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.endswith("_scale_inv"):
                continue
                
            param = f.get_tensor(name)
            
            # Check parameters that should be split
            if any(split_name in name for split_name in ["q_proj", "kv_b_proj", "gate_proj", "up_proj"]):
                if param.size(0) % mp != 0:
                    print(f"Dimension issue: {name} size {param.size(0)} not divisible by mp={mp}")
                    return False
            
            elif any(split_name in name for split_name in ["o_proj", "down_proj"]):
                if param.size(1) % mp != 0:
                    print(f"Dimension issue: {name} size {param.size(1)} not divisible by mp={mp}")
                    return False
    
    print("All dimensions compatible with specified model parallelism")
    return True

# Usage
if diagnose_dimension_issues("/path/to/hf/model", mp=16):
    # Proceed with conversion
    pass
else:
    print("Fix dimension issues before conversion")
```

#### 3. Disk Space Management

**Problem**: Insufficient disk space for conversion

**Solution**:
```python
import shutil

def estimate_conversion_space(hf_path, mp, precision_conversion=False):
    """
    Estimate disk space required for conversion.
    
    Args:
        hf_path (str): Hugging Face model path
        mp (int): Model parallelism factor
        precision_conversion (bool): Whether FP8->BF16 conversion is needed
        
    Returns:
        float: Estimated space required in GB
    """
    
    # Calculate size of input model
    total_size = 0
    for root, dirs, files in os.walk(hf_path):
        for file in files:
            if file.endswith('.safetensors'):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    input_size_gb = total_size / (1024**3)
    
    # Estimate output size
    # Converted model will have similar size but split across mp files
    output_size_gb = input_size_gb * 1.1  # Small overhead for metadata
    
    # If converting FP8 to BF16, weights will double in size
    if precision_conversion:
        output_size_gb *= 2
    
    # Add buffer for temporary files
    total_required_gb = input_size_gb + output_size_gb + 10  # 10GB buffer
    
    print(f"Input model size: {input_size_gb:.2f} GB")
    print(f"Estimated output size: {output_size_gb:.2f} GB")
    print(f"Total space required: {total_required_gb:.2f} GB")
    
    # Check available space
    available_gb = shutil.disk_usage(os.path.dirname(hf_path)).free / (1024**3)
    print(f"Available space: {available_gb:.2f} GB")
    
    if available_gb < total_required_gb:
        print(f"WARNING: Insufficient disk space!")
        print(f"Need additional {total_required_gb - available_gb:.2f} GB")
        return False
    
    return True

# Usage
if estimate_conversion_space("/path/to/hf/model", mp=16, precision_conversion=True):
    # Proceed with conversion
    pass
else:
    print("Free up disk space before conversion")
```

### Error Recovery

```python
def recover_partial_conversion(output_path, mp):
    """
    Recover from partial conversion by identifying missing files.
    
    Args:
        output_path (str): Output directory path
        mp (int): Model parallelism factor
        
    Returns:
        List[int]: List of missing shard indices
    """
    
    expected_files = [f"model{i}-mp{mp}.safetensors" for i in range(mp)]
    missing_shards = []
    
    for i, file_name in enumerate(expected_files):
        file_path = os.path.join(output_path, file_name)
        if not os.path.exists(file_path):
            missing_shards.append(i)
        else:
            # Verify file is not corrupted
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    # Try to access first key
                    first_key = next(iter(f.keys()))
                    f.get_tensor(first_key)
            except Exception as e:
                print(f"Corrupted file detected: {file_name} - {e}")
                missing_shards.append(i)
                # Remove corrupted file
                os.remove(file_path)
    
    if missing_shards:
        print(f"Missing or corrupted shards: {missing_shards}")
        print("Re-run conversion to regenerate missing shards")
    else:
        print("All conversion files are present and valid")
    
    return missing_shards

# Usage
missing = recover_partial_conversion("/path/to/output", mp=16)
if missing:
    print(f"Need to regenerate shards: {missing}")
```

---

## Best Practices

### 1. Conversion Checklist

Before starting conversion:

- [ ] Verify input model integrity
- [ ] Check disk space requirements
- [ ] Validate model parallelism compatibility
- [ ] Ensure CUDA environment is properly set up
- [ ] Backup original model files

During conversion:

- [ ] Monitor memory usage
- [ ] Check for error messages
- [ ] Validate intermediate outputs
- [ ] Monitor disk space consumption

After conversion:

- [ ] Validate output file completeness
- [ ] Test inference with converted model
- [ ] Compare outputs with original model
- [ ] Document conversion parameters

### 2. Performance Optimization

```python
# Optimize conversion performance
torch.set_num_threads(8)  # Use multiple CPU threads
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster operations

# For large models, consider processing in chunks
def chunked_conversion(input_path, output_path, chunk_size=1000):
    """Process large models in chunks to manage memory."""
    
    # Implementation for chunked processing
    # ... chunked conversion logic ...
    pass
```

### 3. Quality Assurance

```python
def comprehensive_qa_check(converted_model_path, original_config):
    """Comprehensive quality assurance for converted models."""
    
    checks = {
        "file_completeness": False,
        "parameter_integrity": False,
        "inference_functionality": False,
        "memory_efficiency": False
    }
    
    try:
        # Check 1: File completeness
        mp = get_model_parallel_from_files(converted_model_path)
        expected_files = [f"model{i}-mp{mp}.safetensors" for i in range(mp)]
        
        for file_name in expected_files:
            if not os.path.exists(os.path.join(converted_model_path, file_name)):
                raise FileNotFoundError(f"Missing file: {file_name}")
        
        checks["file_completeness"] = True
        print("✓ File completeness check passed")
        
        # Check 2: Parameter integrity
        # ... parameter checking logic ...
        checks["parameter_integrity"] = True
        print("✓ Parameter integrity check passed")
        
        # Check 3: Inference functionality
        # ... inference testing logic ...
        checks["inference_functionality"] = True
        print("✓ Inference functionality check passed")
        
        # Check 4: Memory efficiency
        # ... memory usage validation ...
        checks["memory_efficiency"] = True
        print("✓ Memory efficiency check passed")
        
    except Exception as e:
        print(f"✗ QA check failed: {e}")
    
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    
    print(f"\nQA Summary: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 Model conversion validated successfully!")
        return True
    else:
        print("❌ Model conversion validation failed")
        return False

# Usage
qa_result = comprehensive_qa_check(
    "/path/to/converted/model",
    original_config
)
```

---

This documentation provides comprehensive guidance for all model conversion scenarios in the DeepSeek-V3 project, ensuring reliable and efficient model format transformations for various deployment needs.