# DeepSeek-V3 Configuration Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Model Configuration Files](#model-configuration-files)
3. [ModelArgs Parameters](#modelargs-parameters)
4. [Configuration Examples](#configuration-examples)
5. [Environment Variables](#environment-variables)
6. [Runtime Configuration](#runtime-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Deployment Configurations](#deployment-configurations)

---

## Overview

DeepSeek-V3 uses a comprehensive configuration system that controls model architecture, training parameters, inference settings, and deployment options. This guide provides detailed information about all configuration options and their effects.

---

## Model Configuration Files

### Available Configurations

The project includes three pre-defined model configurations:

| Configuration | File | Total Params | Activated Params | Use Case |
|---------------|------|--------------|------------------|----------|
| **16B Model** | `config_16B.json` | 16B | ~8B | Development, Testing |
| **236B Model** | `config_236B.json` | 236B | ~21B | Medium-scale deployment |
| **671B Model** | `config_671B.json` | 671B | ~37B | Production, Full capability |

### Configuration File Format

All configuration files use JSON format with the following structure:

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

## ModelArgs Parameters

### Core Architecture Parameters

#### Basic Model Dimensions

| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|--------|
| `vocab_size` | int | 102400 | Vocabulary size | Memory usage for embeddings |
| `dim` | int | 2048 | Model hidden dimension | Overall model capacity |
| `n_layers` | int | 27 | Number of transformer layers | Model depth and capacity |
| `n_heads` | int | 16 | Number of attention heads | Attention parallelism |

**Configuration Example:**
```python
# Small model for development
small_config = {
    "vocab_size": 50000,
    "dim": 1024,
    "n_layers": 12,
    "n_heads": 16
}

# Large model for production
large_config = {
    "vocab_size": 129280,
    "dim": 7168,
    "n_layers": 61,
    "n_heads": 128
}
```

#### MLP and Feed-Forward Dimensions

| Parameter | Type | Default | Description | Calculation |
|-----------|------|---------|-------------|-------------|
| `inter_dim` | int | 10944 | Intermediate dimension for dense layers | Usually ~2.7 × `dim` |
| `moe_inter_dim` | int | 1408 | Intermediate dimension for MoE experts | Usually ~0.7 × `dim` |

**Sizing Guidelines:**
```python
def calculate_mlp_dimensions(dim):
    """Calculate recommended MLP dimensions based on model dimension."""
    return {
        "inter_dim": int(dim * 2.7),  # Dense layer intermediate dim
        "moe_inter_dim": int(dim * 0.7)  # MoE expert intermediate dim
    }

# Examples
print(calculate_mlp_dimensions(1024))  # {'inter_dim': 2764, 'moe_inter_dim': 716}
print(calculate_mlp_dimensions(4096))  # {'inter_dim': 11059, 'moe_inter_dim': 2867}
```

---

### Mixture-of-Experts (MoE) Parameters

#### Expert Configuration

| Parameter | Type | Default | Description | Constraints |
|-----------|------|---------|-------------|-------------|
| `n_routed_experts` | int | 64 | Total number of routed experts | Must be divisible by `world_size` |
| `n_shared_experts` | int | 2 | Number of always-active experts | Usually 1-2 |
| `n_activated_experts` | int | 6 | Experts activated per token | Usually 10-15% of total |
| `n_expert_groups` | int | 1 | Number of expert groups | For hierarchical routing |
| `n_limited_groups` | int | 1 | Number of limited groups | For load balancing |

#### Routing Configuration

| Parameter | Type | Default | Description | Options |
|-----------|------|---------|-------------|---------|
| `score_func` | str | "softmax" | Scoring function for routing | "softmax", "sigmoid" |
| `route_scale` | float | 1.0 | Scaling factor for routing scores | 1.0-3.0 typical range |

**MoE Configuration Examples:**

```python
# High-capacity MoE (more experts, higher activation)
high_capacity_moe = {
    "n_routed_experts": 256,
    "n_shared_experts": 2,
    "n_activated_experts": 16,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "score_func": "sigmoid",
    "route_scale": 2.5
}

# Memory-efficient MoE (fewer experts, lower activation)
efficient_moe = {
    "n_routed_experts": 64,
    "n_shared_experts": 1,
    "n_activated_experts": 4,
    "n_expert_groups": 2,
    "n_limited_groups": 1,
    "score_func": "softmax",
    "route_scale": 1.0
}

# Calculate expert utilization
def calculate_expert_utilization(config):
    """Calculate expert utilization statistics."""
    total_experts = config["n_routed_experts"] + config["n_shared_experts"]
    activated_experts = config["n_activated_experts"] + config["n_shared_experts"]
    utilization_rate = activated_experts / total_experts
    
    return {
        "total_experts": total_experts,
        "activated_per_token": activated_experts,
        "utilization_rate": f"{utilization_rate:.1%}",
        "routing_efficiency": config["n_activated_experts"] / config["n_routed_experts"]
    }

print(calculate_expert_utilization(high_capacity_moe))
```

---

### Multi-head Latent Attention (MLA) Parameters

#### LoRA Configuration

| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|--------|
| `q_lora_rank` | int | 0 | LoRA rank for query projection | 0 = disabled, >0 = enabled |
| `kv_lora_rank` | int | 512 | LoRA rank for key-value projection | Memory efficiency |

#### Head Dimensions

| Parameter | Type | Default | Description | Calculation |
|-----------|------|---------|-------------|-------------|
| `qk_nope_head_dim` | int | 128 | Non-positional query-key dimension | Usually 128 |
| `qk_rope_head_dim` | int | 64 | Rotary positional dimension | Usually 64 |
| `v_head_dim` | int | 128 | Value head dimension | Usually equals `qk_nope_head_dim` |

**MLA Configuration Examples:**

```python
# High-efficiency MLA (with LoRA)
efficient_mla = {
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128
}

# Standard MLA (without query LoRA)
standard_mla = {
    "q_lora_rank": 0,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128
}

def calculate_mla_memory_savings(config, seq_len, n_heads):
    """Calculate memory savings from MLA compression."""
    
    if config["kv_lora_rank"] == 0:
        return {"savings": "0%", "reason": "No compression"}
    
    # Traditional attention KV memory
    traditional_kv_dim = config["qk_nope_head_dim"] + config["qk_rope_head_dim"] + config["v_head_dim"]
    traditional_memory = seq_len * n_heads * traditional_kv_dim
    
    # MLA compressed memory
    mla_memory = seq_len * (config["kv_lora_rank"] + config["qk_rope_head_dim"])
    
    savings_ratio = (traditional_memory - mla_memory) / traditional_memory
    
    return {
        "traditional_memory_units": traditional_memory,
        "mla_memory_units": mla_memory,
        "savings": f"{savings_ratio:.1%}",
        "compression_ratio": f"{traditional_memory / mla_memory:.1f}x"
    }

print(calculate_mla_memory_savings(efficient_mla, seq_len=4096, n_heads=128))
```

---

### Positional Encoding Parameters (YARN)

#### RoPE Configuration

| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|--------|
| `original_seq_len` | int | 4096 | Base sequence length | YARN scaling reference |
| `rope_theta` | float | 10000.0 | RoPE base frequency | Positional encoding range |
| `rope_factor` | float | 40 | Scaling factor for extension | Context length scaling |
| `beta_fast` | int | 32 | Fast beta correction | High-frequency adjustment |
| `beta_slow` | int | 1 | Slow beta correction | Low-frequency adjustment |
| `mscale` | float | 1.0 | Attention scaling factor | Extended context scaling |

**YARN Configuration Examples:**

```python
# Standard context length (4K)
standard_rope = {
    "original_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_factor": 1.0,  # No extension
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0
}

# Extended context length (128K)
extended_rope = {
    "original_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_factor": 40.0,  # 40x extension
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0
}

def calculate_effective_context_length(config):
    """Calculate effective context length with YARN scaling."""
    
    base_length = config["original_seq_len"]
    extension_factor = config["rope_factor"]
    
    if extension_factor > 1.0:
        effective_length = base_length * extension_factor
        return {
            "base_length": base_length,
            "extension_factor": extension_factor,
            "effective_length": int(effective_length),
            "scaling_method": "YARN"
        }
    else:
        return {
            "base_length": base_length,
            "extension_factor": 1.0,
            "effective_length": base_length,
            "scaling_method": "Standard RoPE"
        }

print(calculate_effective_context_length(extended_rope))
```

---

### Runtime and Inference Parameters

#### Memory and Batch Configuration

| Parameter | Type | Default | Description | Tuning Guide |
|-----------|------|---------|-------------|--------------|
| `max_batch_size` | int | 8 | Maximum batch size | Reduce for memory constraints |
| `max_seq_len` | int | 16384 | Maximum sequence length | Balance memory vs capability |
| `dtype` | str | "bf16" | Computation precision | "bf16" or "fp8" |

#### Dense Layer Configuration

| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|--------|
| `n_dense_layers` | int | 1 | Number of non-MoE layers | Usually first few layers |

**Runtime Configuration Examples:**

```python
# Memory-constrained configuration
memory_efficient = {
    "max_batch_size": 2,
    "max_seq_len": 4096,
    "dtype": "fp8"
}

# High-throughput configuration
high_throughput = {
    "max_batch_size": 16,
    "max_seq_len": 8192,
    "dtype": "bf16"
}

# Development configuration
development = {
    "max_batch_size": 1,
    "max_seq_len": 2048,
    "dtype": "bf16"
}

def estimate_memory_usage(config):
    """Estimate GPU memory usage for a configuration."""
    
    # Rough estimation formulas
    model_params = config.get("dim", 2048) * config.get("n_layers", 27) * 1000  # Simplified
    batch_memory = config["max_batch_size"] * config["max_seq_len"] * config.get("dim", 2048)
    
    # Memory multipliers by dtype
    dtype_multipliers = {"fp8": 1, "bf16": 2, "fp32": 4}
    multiplier = dtype_multipliers.get(config.get("dtype", "bf16"), 2)
    
    total_memory_mb = (model_params + batch_memory) * multiplier * 4 / (1024 * 1024)  # Bytes to MB
    
    return {
        "estimated_memory_gb": total_memory_mb / 1024,
        "dtype": config.get("dtype", "bf16"),
        "batch_size": config["max_batch_size"],
        "sequence_length": config["max_seq_len"]
    }

print(estimate_memory_usage(memory_efficient))
```

---

## Configuration Examples

### 1. Custom Model Sizes

```python
from model import ModelArgs

# Tiny model for experimentation
tiny_config = ModelArgs(
    vocab_size=32000,
    dim=512,
    inter_dim=1376,
    moe_inter_dim=256,
    n_layers=8,
    n_dense_layers=2,
    n_heads=8,
    n_routed_experts=16,
    n_shared_experts=1,
    n_activated_experts=2,
    max_seq_len=1024,
    dtype="bf16"
)

# Medium model for balanced performance
medium_config = ModelArgs(
    vocab_size=65000,
    dim=2048,
    inter_dim=5504,
    moe_inter_dim=1024,
    n_layers=24,
    n_dense_layers=2,
    n_heads=32,
    n_routed_experts=64,
    n_shared_experts=1,
    n_activated_experts=4,
    max_seq_len=8192,
    dtype="bf16"
)

# Large model for maximum capability
large_config = ModelArgs(
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
    max_seq_len=131072,  # 128K context
    dtype="fp8"
)
```

### 2. Specialized Configurations

```python
# Code-optimized configuration
code_config = ModelArgs(
    vocab_size=100000,  # Larger vocab for code tokens
    dim=4096,
    inter_dim=11008,
    n_layers=32,
    n_heads=32,
    n_activated_experts=6,  # More experts for code diversity
    max_seq_len=16384,  # Longer context for code
    dtype="bf16"
)

# Math-optimized configuration
math_config = ModelArgs(
    vocab_size=80000,
    dim=3072,
    inter_dim=8192,
    n_layers=36,
    n_heads=24,
    n_activated_experts=8,  # Higher activation for reasoning
    max_seq_len=8192,
    dtype="bf16"
)

# Chat-optimized configuration
chat_config = ModelArgs(
    vocab_size=120000,
    dim=5120,
    inter_dim=13824,
    n_layers=40,
    n_heads=40,
    n_activated_experts=6,
    max_seq_len=32768,  # Long context for conversations
    dtype="bf16"
)
```

### 3. Resource-Constrained Configurations

```python
# Single GPU configuration
single_gpu_config = ModelArgs(
    vocab_size=50000,
    dim=1024,
    inter_dim=2816,
    moe_inter_dim=512,
    n_layers=16,
    n_dense_layers=4,  # More dense layers for single GPU
    n_heads=16,
    n_routed_experts=32,
    n_shared_experts=2,
    n_activated_experts=4,
    max_batch_size=2,
    max_seq_len=2048,
    dtype="bf16"
)

# CPU-inference configuration (not recommended for large models)
cpu_config = ModelArgs(
    vocab_size=30000,
    dim=768,
    inter_dim=2048,
    moe_inter_dim=384,
    n_layers=12,
    n_dense_layers=12,  # All dense for CPU efficiency
    n_heads=12,
    n_routed_experts=0,  # Disable MoE for CPU
    n_shared_experts=0,
    n_activated_experts=0,
    max_batch_size=1,
    max_seq_len=1024,
    dtype="bf16"
)
```

---

## Environment Variables

### Distributed Training/Inference

| Variable | Purpose | Example | Required |
|----------|---------|---------|----------|
| `WORLD_SIZE` | Total number of processes | `16` | Multi-GPU |
| `RANK` | Global rank of current process | `0` | Multi-GPU |
| `LOCAL_RANK` | Local rank within node | `0` | Multi-GPU |
| `MASTER_ADDR` | Master node address | `192.168.1.100` | Multi-node |
| `MASTER_PORT` | Master node port | `29500` | Multi-node |

### Performance Optimization

| Variable | Purpose | Example | Impact |
|----------|---------|---------|--------|
| `CUDA_VISIBLE_DEVICES` | Visible GPU devices | `0,1,2,3` | GPU selection |
| `OMP_NUM_THREADS` | CPU threads | `8` | CPU performance |
| `TORCH_LOGS` | PyTorch logging | `+dynamo` | Debugging |
| `TORCHDYNAMO_VERBOSE` | Verbose compilation | `1` | Debugging |

### Environment Setup Example

```bash
#!/bin/bash
# setup_environment.sh

# Distributed setup for 2 nodes, 8 GPUs each
export WORLD_SIZE=16
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"

# Performance optimization
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS=8

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Debugging (optional)
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

echo "Environment configured for distributed inference"
```

---

## Runtime Configuration

### Global Settings

The model uses several global variables that can be configured:

```python
# In model.py
world_size = 1          # Set by distributed initialization
rank = 0               # Set by distributed initialization  
block_size = 128       # Quantization block size
gemm_impl = "bf16"     # GEMM implementation ("bf16" or "fp8")
attn_impl = "absorb"   # Attention implementation ("naive" or "absorb")
```

### Configuration at Runtime

```python
import model

def configure_runtime(config_dict):
    """Configure global runtime settings."""
    
    # Set quantization block size
    if "block_size" in config_dict:
        model.block_size = config_dict["block_size"]
    
    # Set GEMM implementation
    if "gemm_impl" in config_dict:
        model.gemm_impl = config_dict["gemm_impl"]
    
    # Set attention implementation
    if "attn_impl" in config_dict:
        model.attn_impl = config_dict["attn_impl"]
    
    print(f"Runtime configured: block_size={model.block_size}, "
          f"gemm_impl={model.gemm_impl}, attn_impl={model.attn_impl}")

# Example configuration
runtime_config = {
    "block_size": 256,      # Larger blocks for better performance
    "gemm_impl": "fp8",     # Use FP8 GEMM for efficiency
    "attn_impl": "absorb"   # Use memory-efficient attention
}

configure_runtime(runtime_config)
```

---

## Performance Tuning

### Memory Optimization

```python
def create_memory_optimized_config(base_config, available_memory_gb):
    """
    Create memory-optimized configuration based on available GPU memory.
    
    Args:
        base_config (dict): Base configuration
        available_memory_gb (float): Available GPU memory in GB
        
    Returns:
        dict: Optimized configuration
    """
    
    config = base_config.copy()
    
    # Adjust batch size based on available memory
    if available_memory_gb < 16:
        config["max_batch_size"] = 1
        config["max_seq_len"] = min(config.get("max_seq_len", 4096), 2048)
    elif available_memory_gb < 32:
        config["max_batch_size"] = 2
        config["max_seq_len"] = min(config.get("max_seq_len", 4096), 4096)
    elif available_memory_gb < 64:
        config["max_batch_size"] = 4
        config["max_seq_len"] = min(config.get("max_seq_len", 8192), 8192)
    
    # Use FP8 for memory efficiency
    if available_memory_gb < 32:
        config["dtype"] = "fp8"
    
    return config

# Usage
optimized_config = create_memory_optimized_config(
    base_config={"dim": 4096, "n_layers": 32},
    available_memory_gb=24
)
```

### Throughput Optimization

```python
def create_throughput_optimized_config(base_config):
    """
    Create configuration optimized for maximum throughput.
    
    Args:
        base_config (dict): Base configuration
        
    Returns:
        dict: Throughput-optimized configuration
    """
    
    config = base_config.copy()
    
    # Optimize for throughput
    config["max_batch_size"] = min(config.get("max_batch_size", 8), 16)  # Larger batches
    config["dtype"] = "fp8"  # Faster computation
    
    # Optimize expert configuration for throughput
    if "n_activated_experts" in config:
        # Slightly reduce activated experts for speed
        config["n_activated_experts"] = max(1, config["n_activated_experts"] - 1)
    
    # Use sigmoid routing for faster computation
    config["score_func"] = "sigmoid"
    
    return config
```

---

## Deployment Configurations

### Production Deployment

```json
{
    "name": "production-config",
    "model": {
        "vocab_size": 129280,
        "dim": 7168,
        "n_layers": 61,
        "n_heads": 128,
        "dtype": "fp8",
        "max_batch_size": 8,
        "max_seq_len": 32768
    },
    "deployment": {
        "model_parallel": 16,
        "pipeline_parallel": 1,
        "tensor_parallel": 16,
        "max_concurrent_requests": 100,
        "timeout_seconds": 300
    },
    "optimization": {
        "use_torch_compile": true,
        "enable_tf32": true,
        "use_flash_attention": true,
        "kv_cache_dtype": "fp8"
    }
}
```

### Development Configuration

```json
{
    "name": "development-config",
    "model": {
        "vocab_size": 50000,
        "dim": 1024,
        "n_layers": 12,
        "n_heads": 16,
        "dtype": "bf16",
        "max_batch_size": 2,
        "max_seq_len": 2048
    },
    "deployment": {
        "model_parallel": 1,
        "pipeline_parallel": 1,
        "tensor_parallel": 1,
        "max_concurrent_requests": 10,
        "timeout_seconds": 60
    },
    "optimization": {
        "use_torch_compile": false,
        "enable_tf32": true,
        "use_flash_attention": false,
        "kv_cache_dtype": "bf16"
    }
}
```

### Configuration Validation

```python
def validate_configuration(config):
    """
    Validate model configuration for consistency and feasibility.
    
    Args:
        config (dict or ModelArgs): Configuration to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    
    errors = []
    
    # Convert to dict if ModelArgs instance
    if hasattr(config, '__dict__'):
        config = config.__dict__
    
    # Validate basic constraints
    if config.get("n_routed_experts", 0) % config.get("n_expert_groups", 1) != 0:
        errors.append("n_routed_experts must be divisible by n_expert_groups")
    
    if config.get("n_activated_experts", 0) > config.get("n_routed_experts", 0):
        errors.append("n_activated_experts cannot exceed n_routed_experts")
    
    if config.get("n_limited_groups", 0) > config.get("n_expert_groups", 1):
        errors.append("n_limited_groups cannot exceed n_expert_groups")
    
    # Validate dimension relationships
    qk_total_dim = config.get("qk_nope_head_dim", 0) + config.get("qk_rope_head_dim", 0)
    if qk_total_dim == 0:
        errors.append("Total query-key dimension must be positive")
    
    # Validate memory constraints
    estimated_memory = estimate_memory_usage(config)
    if estimated_memory["estimated_memory_gb"] > 1000:  # Arbitrary large limit
        errors.append(f"Configuration may require excessive memory: {estimated_memory['estimated_memory_gb']:.1f} GB")
    
    # Validate dtype
    valid_dtypes = ["bf16", "fp8"]
    if config.get("dtype") not in valid_dtypes:
        errors.append(f"Invalid dtype: {config.get('dtype')}. Must be one of {valid_dtypes}")
    
    return len(errors) == 0, errors

# Usage
is_valid, errors = validate_configuration(large_config.__dict__)
if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

---

## Advanced Configuration Patterns

### 1. Dynamic Configuration Loading

```python
import json
import os
from typing import Dict, Any

class ConfigurationManager:
    """Advanced configuration management system."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """Load all configuration templates."""
        for file_name in os.listdir(self.config_dir):
            if file_name.endswith('.json'):
                template_name = file_name.replace('.json', '').replace('config_', '')
                
                with open(os.path.join(self.config_dir, file_name)) as f:
                    self.templates[template_name] = json.load(f)
    
    def create_custom_config(self, base_template: str, modifications: Dict[str, Any]) -> ModelArgs:
        """
        Create custom configuration based on template.
        
        Args:
            base_template (str): Name of base template
            modifications (dict): Parameters to modify
            
        Returns:
            ModelArgs: Custom configuration
        """
        
        if base_template not in self.templates:
            raise ValueError(f"Template '{base_template}' not found. Available: {list(self.templates.keys())}")
        
        # Start with base template
        config = self.templates[base_template].copy()
        
        # Apply modifications
        config.update(modifications)
        
        # Validate configuration
        is_valid, errors = validate_configuration(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        return ModelArgs(**config)
    
    def get_recommended_config(self, use_case: str, available_gpus: int = 1) -> ModelArgs:
        """
        Get recommended configuration for specific use case.
        
        Args:
            use_case (str): Use case ("development", "production", "research")
            available_gpus (int): Number of available GPUs
            
        Returns:
            ModelArgs: Recommended configuration
        """
        
        if use_case == "development":
            base = "16B" if available_gpus == 1 else "236B"
            modifications = {
                "max_batch_size": 2,
                "max_seq_len": 2048,
                "dtype": "bf16"
            }
        
        elif use_case == "production":
            base = "671B" if available_gpus >= 8 else "236B"
            modifications = {
                "max_batch_size": 8,
                "max_seq_len": 16384,
                "dtype": "fp8" if available_gpus >= 8 else "bf16"
            }
        
        elif use_case == "research":
            base = "236B"
            modifications = {
                "max_batch_size": 4,
                "max_seq_len": 8192,
                "dtype": "bf16"
            }
        
        else:
            raise ValueError(f"Unknown use case: {use_case}")
        
        return self.create_custom_config(base, modifications)

# Usage
config_manager = ConfigurationManager()

# Get recommended configuration
dev_config = config_manager.get_recommended_config("development", available_gpus=1)
prod_config = config_manager.get_recommended_config("production", available_gpus=16)

# Create custom configuration
custom_config = config_manager.create_custom_config("671B", {
    "max_seq_len": 65536,  # 64K context
    "dtype": "bf16",
    "temperature": 0.5
})
```

### 2. Configuration Profiles

```python
class ConfigurationProfiles:
    """Predefined configuration profiles for common scenarios."""
    
    @staticmethod
    def get_inference_profile(model_size: str = "671B") -> Dict[str, Any]:
        """Get optimized configuration for inference."""
        
        base_configs = {
            "16B": {"dim": 2048, "n_layers": 27, "n_heads": 16},
            "236B": {"dim": 5120, "n_layers": 48, "n_heads": 40},
            "671B": {"dim": 7168, "n_layers": 61, "n_heads": 128}
        }
        
        if model_size not in base_configs:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = base_configs[model_size].copy()
        config.update({
            "dtype": "fp8",
            "max_batch_size": 8,
            "max_seq_len": 16384,
            # Inference optimizations
            "use_kv_cache": True,
            "compile_model": True
        })
        
        return config
    
    @staticmethod
    def get_training_profile(model_size: str = "236B") -> Dict[str, Any]:
        """Get optimized configuration for training."""
        
        config = ConfigurationProfiles.get_inference_profile(model_size)
        config.update({
            "max_batch_size": 4,  # Smaller batch for training
            "max_seq_len": 4096,  # Shorter sequences for memory
            "gradient_checkpointing": True,
            "use_flash_attention": True
        })
        
        return config
    
    @staticmethod
    def get_evaluation_profile(model_size: str = "671B") -> Dict[str, Any]:
        """Get optimized configuration for evaluation."""
        
        config = ConfigurationProfiles.get_inference_profile(model_size)
        config.update({
            "max_batch_size": 1,  # Single sample for accurate evaluation
            "temperature": 0.0,   # Deterministic generation
            "max_new_tokens": 512
        })
        
        return config

# Usage
inference_config = ConfigurationProfiles.get_inference_profile("671B")
training_config = ConfigurationProfiles.get_training_profile("236B")
eval_config = ConfigurationProfiles.get_evaluation_profile("671B")
```

---

## Configuration Best Practices

### 1. Parameter Selection Guidelines

#### Model Dimension (`dim`)
- **Small models (< 10B)**: 1024-2048
- **Medium models (10B-100B)**: 2048-5120
- **Large models (> 100B)**: 5120-7168+

#### Expert Configuration
- **Expert ratio**: `n_activated_experts` should be 10-15% of `n_routed_experts`
- **Shared experts**: Usually 1-2, always active
- **Expert groups**: Use for hierarchical routing in very large models

#### Sequence Length
- **Development**: 1024-2048 tokens
- **Production**: 8192-32768 tokens  
- **Long context**: 65536-131072 tokens (with YARN scaling)

### 2. Hardware-Specific Configurations

```python
def get_hardware_optimized_config(gpu_type: str, gpu_count: int) -> Dict[str, Any]:
    """
    Get configuration optimized for specific hardware.
    
    Args:
        gpu_type (str): Type of GPU ("A100", "H100", "V100", etc.)
        gpu_count (int): Number of available GPUs
        
    Returns:
        dict: Hardware-optimized configuration
    """
    
    # Base configurations by GPU type
    gpu_configs = {
        "V100": {
            "dtype": "bf16",  # V100 doesn't support FP8
            "max_batch_size": 2,
            "block_size": 128
        },
        "A100": {
            "dtype": "fp8",   # A100 supports FP8
            "max_batch_size": 4,
            "block_size": 128
        },
        "H100": {
            "dtype": "fp8",   # H100 optimized for FP8
            "max_batch_size": 8,
            "block_size": 256
        }
    }
    
    if gpu_type not in gpu_configs:
        # Default to conservative settings
        gpu_type = "V100"
    
    config = gpu_configs[gpu_type].copy()
    
    # Adjust for GPU count
    if gpu_count >= 8:
        config["max_batch_size"] *= 2
        config["max_seq_len"] = 16384
    elif gpu_count >= 4:
        config["max_seq_len"] = 8192
    else:
        config["max_seq_len"] = 4096
    
    return config

# Usage
h100_config = get_hardware_optimized_config("H100", gpu_count=8)
a100_config = get_hardware_optimized_config("A100", gpu_count=4)
```

### 3. Use Case Specific Configurations

```python
class UseCaseConfigs:
    """Configuration templates for specific use cases."""
    
    @staticmethod
    def chatbot_config():
        """Configuration optimized for chatbot applications."""
        return {
            "max_seq_len": 32768,  # Long context for conversations
            "max_batch_size": 4,   # Moderate batch size
            "temperature": 0.7,    # Balanced creativity
            "top_p": 0.9,         # Nucleus sampling
            "repetition_penalty": 1.1,
            "dtype": "bf16"
        }
    
    @staticmethod
    def code_generation_config():
        """Configuration optimized for code generation."""
        return {
            "max_seq_len": 16384,  # Sufficient for code files
            "max_batch_size": 2,   # Smaller batch for code quality
            "temperature": 0.2,    # Lower temperature for precision
            "top_p": 0.95,
            "dtype": "bf16",
            # Code-specific optimizations
            "stop_sequences": ["```", "\n\n\n"]
        }
    
    @staticmethod
    def research_config():
        """Configuration optimized for research applications."""
        return {
            "max_seq_len": 8192,   # Balanced context length
            "max_batch_size": 1,   # Single sample for reproducibility
            "temperature": 0.0,    # Deterministic output
            "dtype": "bf16",
            "seed": 42,           # Fixed seed for reproducibility
            "output_scores": True  # Return token probabilities
        }

# Usage
chatbot_cfg = UseCaseConfigs.chatbot_config()
code_cfg = UseCaseConfigs.code_generation_config()
research_cfg = UseCaseConfigs.research_config()
```

---

## Configuration Migration and Versioning

### Version Compatibility

```python
def migrate_config_version(old_config: Dict[str, Any], target_version: str) -> Dict[str, Any]:
    """
    Migrate configuration between versions.
    
    Args:
        old_config (dict): Configuration from older version
        target_version (str): Target version ("v3.0", "v3.1", etc.)
        
    Returns:
        dict: Migrated configuration
    """
    
    new_config = old_config.copy()
    
    if target_version == "v3.0":
        # Add new parameters with defaults
        new_config.setdefault("route_scale", 1.0)
        new_config.setdefault("score_func", "softmax")
        new_config.setdefault("mscale", 1.0)
        
        # Remove deprecated parameters
        deprecated_params = ["old_param_1", "old_param_2"]
        for param in deprecated_params:
            new_config.pop(param, None)
    
    return new_config

# Usage
old_config = {"dim": 4096, "n_layers": 32}  # Old format
new_config = migrate_config_version(old_config, "v3.0")
```

### Configuration Templates

```python
def create_config_from_template(template_name: str, **overrides) -> ModelArgs:
    """
    Create configuration from predefined template.
    
    Args:
        template_name (str): Template name
        **overrides: Parameter overrides
        
    Returns:
        ModelArgs: Configuration instance
    """
    
    templates = {
        "tiny": {
            "vocab_size": 32000, "dim": 512, "n_layers": 6, "n_heads": 8,
            "n_routed_experts": 8, "n_activated_experts": 2
        },
        "small": {
            "vocab_size": 50000, "dim": 1024, "n_layers": 12, "n_heads": 16,
            "n_routed_experts": 32, "n_activated_experts": 4
        },
        "medium": {
            "vocab_size": 80000, "dim": 2048, "n_layers": 24, "n_heads": 32,
            "n_routed_experts": 64, "n_activated_experts": 6
        },
        "large": {
            "vocab_size": 100000, "dim": 4096, "n_layers": 32, "n_heads": 64,
            "n_routed_experts": 128, "n_activated_experts": 8
        },
        "xlarge": {
            "vocab_size": 129280, "dim": 7168, "n_layers": 61, "n_heads": 128,
            "n_routed_experts": 256, "n_activated_experts": 8
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Template '{template_name}' not found. Available: {list(templates.keys())}")
    
    config = templates[template_name].copy()
    config.update(overrides)
    
    return ModelArgs(**config)

# Usage examples
tiny_model = create_config_from_template("tiny", dtype="bf16")
custom_model = create_config_from_template("medium", max_seq_len=16384, dtype="fp8")
```

---

## Configuration Testing and Validation

### Automated Configuration Testing

```python
def test_configuration_compatibility(config: ModelArgs, available_memory_gb: float) -> Dict[str, Any]:
    """
    Test configuration compatibility with available resources.
    
    Args:
        config (ModelArgs): Configuration to test
        available_memory_gb (float): Available GPU memory
        
    Returns:
        dict: Compatibility test results
    """
    
    results = {
        "memory_compatible": False,
        "performance_rating": "unknown",
        "recommendations": []
    }
    
    # Estimate memory requirements
    estimated_memory = estimate_memory_usage(config.__dict__)
    memory_required = estimated_memory["estimated_memory_gb"]
    
    # Memory compatibility
    if memory_required <= available_memory_gb * 0.8:  # 80% utilization max
        results["memory_compatible"] = True
        results["memory_utilization"] = f"{memory_required / available_memory_gb:.1%}"
    else:
        results["recommendations"].append(
            f"Reduce memory usage: requires {memory_required:.1f}GB, available {available_memory_gb:.1f}GB"
        )
    
    # Performance rating
    if config.dtype == "fp8" and memory_required <= available_memory_gb * 0.6:
        results["performance_rating"] = "excellent"
    elif config.dtype == "bf16" and memory_required <= available_memory_gb * 0.7:
        results["performance_rating"] = "good"
    else:
        results["performance_rating"] = "limited"
        results["recommendations"].append("Consider using FP8 dtype for better performance")
    
    # Specific recommendations
    if config.max_batch_size > 8:
        results["recommendations"].append("Large batch size may impact latency")
    
    if config.max_seq_len > 32768:
        results["recommendations"].append("Very long sequences may impact performance")
    
    return results

# Usage
test_results = test_configuration_compatibility(large_config, available_memory_gb=80)
print(f"Memory compatible: {test_results['memory_compatible']}")
print(f"Performance rating: {test_results['performance_rating']}")
for rec in test_results['recommendations']:
    print(f"Recommendation: {rec}")
```

---

## Quick Configuration Reference

### Essential Parameters Quick Reference

```python
# Minimal working configuration
minimal_config = {
    "vocab_size": 32000,
    "dim": 1024,
    "n_layers": 12,
    "n_heads": 16
}

# Production-ready configuration
production_config = {
    "vocab_size": 129280,
    "dim": 7168,
    "n_layers": 61,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_activated_experts": 8,
    "dtype": "fp8",
    "max_seq_len": 32768
}

# Memory-optimized configuration
memory_optimized_config = {
    "max_batch_size": 2,
    "max_seq_len": 4096,
    "dtype": "fp8",
    "kv_lora_rank": 256,  # Reduce KV memory
    "q_lora_rank": 512    # Reduce query memory
}
```

### Command Line Configuration

```bash
# Environment variables for distributed setup
export WORLD_SIZE=16
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"

# PyTorch optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Launch with custom configuration
torchrun --nnodes 2 --nproc-per-node 8 generate.py \
  --ckpt-path /path/to/model \
  --config custom_config.json \
  --max-new-tokens 200 \
  --temperature 0.7
```

---

*This configuration guide provides comprehensive coverage of all configuration options in the DeepSeek-V3 system. Use it as a reference for optimizing your specific deployment and use case requirements.*