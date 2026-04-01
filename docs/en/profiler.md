# Profiler System

TeleFuser provides a profiling system for performance analysis and debugging, built on PyTorch Profiler with additional features for distributed environments.

## Features

- **Context manager and decorator** - Flexible usage patterns
- **Sync and async support** - Works with both synchronous and asynchronous functions
- **Distributed aware** - Automatically handles multi-rank profiling
- **Memory tracking** - Peak memory allocation monitoring
- **PyTorch Profiler integration** - Optional detailed kernel-level profiling
- **Chrome trace export** - Visualize traces in Chrome DevTools
- **Environment variable control** - Easy enable/disable without code changes

## Quick Start

### Basic Usage

```python
from telefuser.utils.profiler import ProfilingContext

# As context manager
with ProfilingContext("my_operation"):
    # Your code here
    result = model(input_data)

# As decorator
@ProfilingContext("my_function")
def process_data(data):
    return model(data)

# As async decorator
@ProfilingContext("async_operation")
async def process_async(data):
    return await model(data)
```

### Enable PyTorch Profiler

Set environment variables to enable detailed profiling:

```bash
# Enable profiler for specific names
export ENABLE_PROFILER_NAMES="vae_decode,text_encoding,dit_denoising"

# Set output directory for trace files
export PROFILER_OUTPUT_DIR="./profiler_output"

# Run your application
python your_script.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_PROFILER_NAMES` | Comma-separated list of profiler names to enable | "" (empty) |
| `PROFILER_OUTPUT_DIR` | Directory for Chrome trace output files | "./profiler_output" |
| `TELEFUSER_PROFILE_DEBUG` | Enable all debug profiling contexts | "false" |

### Controlling Profiler Activation

```python
from telefuser.utils.profiler import (
    enable_profiler_for_names,
    set_profiler_output_dir,
    get_enabled_profiler_names,
)

# Programmatically enable profilers
enable_profiler_for_names("vae_decode,text_encoding")

# Set output directory
set_profiler_output_dir("/path/to/traces")

# Check enabled names
names = get_enabled_profiler_names()  # Returns: {"vae_decode", "text_encoding"}
```

## ProfilingContext vs ProfilingContext4Debug

### ProfilingContext

Always active profiling context:

```python
from telefuser.utils.profiler import ProfilingContext

@ProfilingContext("operation_name")
def process():
    # Always logs execution time and peak memory
    pass
```

### ProfilingContext4Debug

Conditionally active based on `TELEFUSER_PROFILE_DEBUG`:

```python
from telefuser.utils.profiler import ProfilingContext4Debug

@ProfilingContext4Debug("debug_operation")
def process():
    # Only profiles when TELEFUSER_PROFILE_DEBUG=true
    # Otherwise, no overhead
    pass
```

**Recommended usage in Stage:**

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.utils.profiler import ProfilingContext4Debug

class MyStage(BaseStage):
    @with_model_offload(["model"])
    @ProfilingContext4Debug("my_stage_process")
    @torch.inference_mode()
    def process(self, input_data):
        # Profiling only active in debug mode
        return self.model(input_data)
```

## Output

### Console Logs

When using `ProfilingContext`, the following information is logged:

```
[Profile] my_operation cost 0.123456 seconds
Rank 0 - Function 'my_operation' Peak Memory: 4.50 GB
```

When PyTorch Profiler is enabled:

```
Rank 0 - Starting PyTorch profiler for 'my_operation'
Rank 0 - PyTorch profiler trace saved to: ./profiler_output/my_operation_rank0_run1.json
Rank 0 - Profiler summary for 'my_operation': Total operations: 150
Rank 0 -   1. aten::addmm: CPU=12.34 ms, CUDA=8.56 ms
Rank 0 -   2. aten::copy_: CPU=5.23 ms, CUDA=3.21 ms
...
```

### Chrome Trace Files

When PyTorch Profiler is enabled, Chrome trace files are exported:

```
profiler_output/
├── vae_decode_rank0_run1.json
├── vae_decode_rank0_run2.json
├── text_encoding_rank0_run1.json
└── dit_denoising_rank0_run1.json
```

To visualize:

1. Open Chrome DevTools (`chrome://tracing`)
2. Click "Load" and select the JSON trace file
3. Analyze kernel timing, memory operations, and CPU/GPU timeline

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Profiler name for identification |
| `reset_peak_memory` | bool | True | Reset peak memory stats before profiling |

```python
# Custom memory tracking behavior
with ProfilingContext("operation", reset_peak_memory=False):
    # Peak memory not reset - captures accumulated peak
    pass
```

## Distributed Support

Profiler automatically handles distributed environments:

```python
# In distributed setting (e.g., 2 GPUs)
with ProfilingContext("distributed_op"):
    # Rank 0 logs: "Rank 0 - Function 'distributed_op' Peak Memory: 4.50 GB"
    # Rank 1 logs: "Rank 1 - Function 'distributed_op' Peak Memory: 4.50 GB"
    pass
```

Trace files include rank information:

```
profiler_output/
├── operation_rank0_run1.json
├── operation_rank1_run1.json
```

## Platform Support

Profiler supports multiple hardware platforms:

| Platform | Profiler Activity |
|----------|-------------------|
| CUDA (NVIDIA) | `torch.profiler.ProfilerActivity.CUDA` |
| XPU (Intel) | `torch.profiler.ProfilerActivity.XPU` |
| NPU (Huawei) | `torch.profiler.ProfilerActivity.PrivateUse1` |
| CPU | `torch.profiler.ProfilerActivity.CPU` (always) |

## Integration in Stages

### Typical Usage Pattern

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.utils.profiler import ProfilingContext4Debug
import torch

class VAEDecodeStage(BaseStage):
    def __init__(self, name, module_manager, model_runtime_config):
        super().__init__(name, model_runtime_config)
        self.vae = module_manager.fetch_module("vae")
        self.model_names = ["vae"]

    @with_model_offload(["vae"])
    @ProfilingContext4Debug("vae_decode")
    @torch.inference_mode()
    def process(self, latents):
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            return self.vae.decode(latents)
```

### Profiling Multiple Operations

```python
class TextEncodingStage(BaseStage):
    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    def encode_text(self, prompts):
        # Overall encoding profiled
        with ProfilingContext4Debug("tokenization"):
            tokens = self.tokenizer(prompts)
        with ProfilingContext4Debug("embedding"):
            embeddings = self.text_encoder(tokens)
        return embeddings
```

## Best Practices

### 1. Use Meaningful Names

```python
# Good - descriptive and unique
@ProfilingContext4Debug("vae_decode_video")
@ProfilingContext4Debug("dit_denoising_step_0")

# Avoid - generic or duplicate
@ProfilingContext4Debug("process")
@ProfilingContext4Debug("model")
```

### 2. Use ProfilingContext4Debug in Stages

```python
# Recommended - no overhead in production
@ProfilingContext4Debug("stage_name")
def process(self, data):
    pass

# Avoid in production code - always active
@ProfilingContext("stage_name")
def process(self, data):
    pass
```

### 3. Combine with Other Decorators

Order matters - profiler should wrap the actual computation:

```python
@with_model_offload(["model"])      # Outer: handles model loading
@ProfilingContext4Debug("process")  # Middle: profiles computation
@torch.inference_mode()             # Inner: disables gradients
def process(self, data):
    return self.model(data)
```

### 4. Enable Specifically

```bash
# Enable only what you need
export ENABLE_PROFILER_NAMES="vae_decode"

# Avoid enabling everything (large trace files)
export ENABLE_PROFILER_NAMES="*"  # Not recommended
```

### 5. Use reset_peak_memory Appropriately

```python
# Reset for each independent operation
with ProfilingContext("independent_op", reset_peak_memory=True):
    pass

# Don't reset when tracking accumulated memory
with ProfilingContext("sequence_op", reset_peak_memory=False):
    pass
```

## Troubleshooting

### Large Trace Files

If trace files are too large:

1. Enable only specific profiler names
2. Reduce profiling duration
3. Profile fewer operations

```bash
export ENABLE_PROFILER_NAMES="dit_denoising"  # Only one operation
```

### Missing GPU Activity

If GPU activity is not recorded:

1. Verify platform is supported (CUDA, XPU, NPU)
2. Check CUDA synchronization is working

```python
from telefuser.platforms import current_platform
print(current_platform.device_type)  # Should be "cuda", "xpu", or "npu"
```

### Memory Stats Not Accurate

Ensure synchronization before profiling:

```python
# Profiler automatically syncs, but manual sync for custom timing
from telefuser.platforms import current_platform
current_platform.synchronize()
with ProfilingContext("operation"):
    pass
```

## Related Documentation

- [Adding New Stage](./adding_new_stage.md) - Stage development with profiler integration
- [Metrics](./metrics.md) - Production monitoring and observability
- [Logging](./logging.md) - Logging configuration and usage
- [Configuration](./configuration.md) - Runtime configuration options