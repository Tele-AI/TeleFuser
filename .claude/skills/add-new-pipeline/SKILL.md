---
name: add-new-pipeline
description: Guide for integrating external project pipelines into TeleFuser. Six-phase workflow with interactive checkpoints.
---

# Add New Pipeline Integration Guide

## Trigger Conditions

- User requests to integrate a new model/pipeline from external project
- User mentions "integrate xxx into telefuser"

---

## Workflow Overview

```
Phase 1 → Phase 2.1 → Phase 2.2 → Phase 3 → Phase 4 → Phase 5
 Analyze   Pipeline    Stages     Models   Cleanup    Review
    ↓         ↓          ↓          ↓         ↓         ↓
Checkpoint Checkpoint Checkpoint Checkpoint Checkpoint  Done
```

**Each phase ends with AskUserQuestion checkpoint - wait for approval before proceeding.**

---

## Phase 1: Analyze Original Pipeline

### Goals
1. Understand model architecture by reading source code
2. Document pipeline logic and inference flow
3. Create analysis reports

### Key Tasks

1. **Read pipeline entry point** - Trace `__call__` method execution flow
2. **Read model definitions** - Go deep to actual class implementations (DiT, VAE, Text Encoder)
3. **Create analysis reports** in `examples/<model_name>/analysis/`:
   - `PIPELINE_LOGIC.md` - Entry point, execution steps, key functions
   - `MODEL_DEFINITION.md` - Architecture, configuration, class hierarchy
   - `INFERENCE_LOGIC.md` - Forward flow, data transformations

### Progress Tracking

Create `examples/<model_name>/PROGRESS.md`:

```markdown
# [Model Name] Integration Progress

## Overview
- **Model**: [Name]
- **Type**: [T2V/I2V/T2I/SR]
- **Started**: [Date]

## Phase Status
| Phase | Status | Notes |
|-------|--------|-------|
| 1. Analyze | 🔄 In Progress | |
| 2.1 Pipeline | ⏳ Pending | |
| 2.2 Stages | ⏳ Pending | |
| 3. Integrate | ⏳ Pending | |
| 4. Cleanup | ⏳ Pending | |
| 5. Review | ⏳ Pending | |

## Key Findings
- Architecture patterns: ...
- Special handling required: ...
- Implementation challenges: ...
```

### Model Source Rules

| Component | Integration Method |
|-----------|-------------------|
| **DiT/Transformer** | Source-level (`telefuser/models/<model>_dit.py`, inherit `BaseModel`) |
| VAE | `module_manager.load_from_huggingface()` |
| Text Encoder | `module_manager.load_from_huggingface()` |
| Scheduler | Use existing or HuggingFace |

### 🛑 Phase 1 Checkpoint

After completion:
1. Show analysis report summaries
2. Highlight critical findings (unique patterns, challenges)
3. **AskUserQuestion**: "Phase 1 complete. Ready for Phase 2.1?"

---

## Phase 2.1: Minimal Pipeline Integration (Faithful Copy)

### Goals
1. Create Pipeline class with **faithful copy** of original pipeline logic
2. Initialize models externally using ModuleManager
3. Verify pipeline can be instantiated and run

### ⚠️ CRITICAL: Faithful Copy Requirements for Pipeline

**Same rules as model integration apply to pipeline code:**

| Prohibited | Example |
|------------|---------|
| ❌ Modify any logic | Change computation order |
| ❌ Add/remove operations | Add preprocessing steps |
| ❌ Change parameter names | Rename `num_frames` to `frame_num` |
| ❌ "Optimize" code | Refactor loops, merge functions |

| Allowed | Description |
|---------|-------------|
| ✅ Change inheritance | Inherit `BasePipeline` |
| ✅ Add type annotations | Parameter and return types |
| ✅ Adjust imports | Use TeleFuser imports |
| ✅ Use ModuleManager | `self.dit = module_manager.fetch_module("dit")` |

### Files to Create

```
telefuser/pipelines/<model_name>/
├── __init__.py
└── pipeline.py          # Pipeline class - faithful copy of original
```

### Pipeline Template

```python
# telefuser/pipelines/<model_name>/pipeline.py
from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.module_manager import ModuleManager

class MyModelPipeline(BasePipeline):
    """Pipeline for MyModel - faithful copy from original project."""
    
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        # Division factors for resolution
        self.height_division_factor = 16
        self.width_division_factor = 16
    
    def init(self, module_manager: ModuleManager, config: MyModelConfig):
        """Initialize pipeline with external modules."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        
        # Fetch modules from ModuleManager (added externally)
        self.dit = module_manager.fetch_module("dit")
        self.vae = module_manager.fetch_module("vae")
        self.text_encoder = module_manager.fetch_module("text_encoder")
    
    def __call__(self, prompt: str, ...):
        """Forward pass - FAITHFUL COPY of original pipeline logic.
        
        DO NOT modify:
        - Computation order
        - Parameter names
        - Math formulas
        - Control flow
        """
        # Copy original __call__ logic exactly
        ...
```

### Example File Template

Create `examples/<model_name>/<model>_<task>_<hardware>.py`:

```python
"""Example for MyModel pipeline integration.

This example shows how to:
1. Initialize models externally
2. Add them to ModuleManager
3. Create and run the pipeline
"""

import torch
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.<model_name> import MyModelPipeline, MyModelConfig

PPL_CONFIG = dict(
    name="<model>_<task>_<hardware>",
    num_inference_steps=50,
    cfg_scale=4.0,
)


def get_pipeline(
    model_root: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Initialize pipeline with external model loading."""
    
    # 1. Create ModuleManager
    mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")
    
    # 2. Load models EXTERNALLY (not inside pipeline)
    # DiT - source-level model
    dit_path = f"{model_root}/dit.safetensors"
    mm.load_model(dit_path, name="dit", torch_dtype=torch_dtype)
    
    # VAE - HuggingFace loading
    vae_path = f"{model_root}/vae"
    mm.load_from_huggingface(
        vae_path,
        module_source="diffusers",
        module_class=AutoencoderKL,
        module_name="vae",
    )
    
    # Text Encoder
    text_encoder_path = f"{model_root}/text_encoder"
    mm.load_from_huggingface(
        text_encoder_path,
        module_source="transformers",
        module_class=T5EncoderModel,
        module_name="text_encoder",
    )
    
    # 3. Create pipeline
    pipeline = MyModelPipeline(device=device, torch_dtype=torch_dtype)
    
    # 4. Initialize pipeline with external modules
    config = MyModelConfig()
    pipeline.init(mm, config)
    
    return pipeline


def run(pipeline, prompt: str, ...):
    """Run inference."""
    with torch.inference_mode():
        output = pipeline(prompt=prompt, ...)
    return output


@click.command()
@click.option("--model_root", required=True)
@click.option("--prompt", required=True)
def main(model_root, prompt):
    pipeline = get_pipeline(model_root)
    output = run(pipeline, prompt)
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
```

### Verification

After creating pipeline:

```bash
# Compare original pipeline with integrated version
diff -u <original_pipeline.py> telefuser/pipelines/<model>/pipeline.py

# Ensure only allowed differences:
# - import statements
# - class inheritance (BasePipeline)
# - type annotations
# - ModuleManager.fetch_module() calls
```

### 🛑 Phase 2.1 Checkpoint

After completion:
1. Show `pipeline.py` - highlight it's a faithful copy
2. Show example file with external model initialization
3. **Run diff comparison**
4. **AskUserQuestion**: "Phase 2.1 complete. Pipeline is faithful copy. Ready for Phase 2.2?"

---

## Phase 2.2: Split Pipeline into Stages (Faithful Copy)

### Goals
1. Split pipeline `__call__` into separate Stage classes
2. Each stage inherits `BaseStage`
3. **NO logic modification** - just code organization

### ⚠️ CRITICAL: Stage Splitting Rules

**Stages are for code organization, NOT refactoring:**

| Prohibited | Reason |
|------------|--------|
| ❌ Change computation order | Breaks correctness |
| ❌ Add/remove operations | Breaks correctness |
| ❌ Modify stage interfaces | Breaks data flow |
| ❌ "Optimize" within stages | Introduces bugs |

| Allowed | Description |
|---------|-------------|
| ✅ Group related operations | e.g., all text encoding in one stage |
| ✅ Use BaseStage decorators | `@with_model_offload`, `@torch.inference_mode` |
| ✅ Pass data between stages | Via method parameters |

### Files Structure

```
telefuser/pipelines/<model_name>/
├── __init__.py
├── pipeline.py           # Pipeline class (updated to use stages)
├── text_encoding.py      # Text encoding stage
├── vae.py                # VAE encode/decode stage
├── denoising.py          # DiT denoising stage
└── <other>_stage.py      # Other stages as needed
```

### Stage Template

```python
# telefuser/pipelines/<model_name>/text_encoding.py
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.module_manager import ModuleManager

class TextEncodingStage(BaseStage):
    """Text encoding stage - faithful copy from original pipeline."""
    
    def __init__(self, name: str, module_manager: ModuleManager, config):
        super().__init__(name, config)
        self.text_encoder = module_manager.fetch_module("text_encoder")
        self.model_names = ["text_encoder"]
    
    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def process(self, prompts: list[str]) -> torch.Tensor:
        """Encode text prompts.
        
        FAITHFUL COPY of original encoding logic - DO NOT MODIFY.
        """
        # Copy exact logic from original pipeline
        text_embeddings = self.text_encoder(prompts)
        return text_embeddings
```

### Updated Pipeline Using Stages

```python
# telefuser/pipelines/<model_name>/pipeline.py
class MyModelPipeline(BasePipeline):
    
    def init(self, module_manager: ModuleManager, config: MyModelConfig):
        self._model_info = module_manager.get_model_info()
        self.config = config
        
        # Create stages
        self.text_encoding_stage = TextEncodingStage(
            "text_encoding", module_manager, config.text_encoding_config
        )
        self.vae_stage = VAEStage(
            "vae", module_manager, config.vae_config
        )
        self.denoising_stage = DenoisingStage(
            "denoising", module_manager, config.dit_config
        )
    
    def __call__(self, prompt: str, ...):
        """Forward pass using stages - SAME LOGIC as original."""
        # Stage 1: Text encoding
        text_embeddings = self.text_encoding_stage.process([prompt])
        
        # Stage 2: VAE encoding
        latents = self.vae_stage.encode(image)
        
        # Stage 3: Denoising
        latents = self.denoising_stage.process(latents, text_embeddings, ...)
        
        # Stage 4: VAE decoding
        output = self.vae_stage.decode(latents)
        
        return output
```

### Verification Checklist

After splitting into stages:

```markdown
| Check Item | Pass? |
|------------|-------|
| Computation order identical | □ |
| All operations preserved | □ |
| Parameter passing correct | □ |
| Output matches original | □ |
```

### 🛑 Phase 2.2 Checkpoint

After completion:
1. Show all stage files
2. Show updated pipeline.py
3. **Run verification checklist**
4. **AskUserQuestion**: "Phase 2.2 complete. Stages are faithful copies. Ready for Phase 3?"

---

## Phase 3: Integrate Internal Models

### Goals
1. Implement DiT model at source-level (inherit `BaseModel`)
2. Implement `state_dict_converter()` for loading pretrained weights
3. Verify model loads correctly

### ⚠️ CRITICAL: Faithful Copy Requirements

**When integrating model code from external projects, strictly follow these rules:**

#### 1. Prohibited Modifications

| Prohibited Operation | Example | Reason |
|---------------------|---------|--------|
| ❌ Add/remove tensor operations | `x.flatten(2)`, `x.view()` | Changes data flow |
| ❌ Change parameter passing | `e0[0]` instead of `e0` | Changes parameter meaning |
| ❌ Modify math formulas | `x = x + y` → `x = x + y * e` | Incorrect computation logic |
| ❌ Merge/split functions | Combine multiple attention | Changes semantics |
| ❌ "Optimize" code structure | Refactor, simplify | Introduces bugs |
| ❌ Modify logic branches | Change conditionals | Inconsistent behavior |

#### 2. Allowed Modifications

| Allowed Operation | Description |
|-------------------|-------------|
| ✅ Change inheritance | `ModelMixin` → `BaseModel` |
| ✅ Add type annotations | `def forward(x)` → `def forward(x: torch.Tensor)` |
| ✅ Adjust import paths | Relative → Absolute imports |
| ✅ Remove external dependencies | Like `diffusers`, `transformers` mixin classes |
| ✅ Add `state_dict_converter()` | Weight loading adaptation |
| ✅ Add docstrings | English comments |

#### 3. Verification Steps

**After copying each class, MUST execute:**

```bash
# Compare original and integrated file differences
diff -u <original_file> <integrated_file> | grep "^[-+]" | grep -v "^[-+][-+][-+]"

# Ensure only these types of differences:
# - import statements
# - class inheritance declaration
# - type annotations
# - docstrings
```

**Difference Verification Checklist:**

```markdown
| Check Item | Pass? |
|------------|-------|
| forward() logic identical | □ |
| Parameter names and order identical | □ |
| Tensor operation calls identical | □ |
| Math formulas identical | □ |
| Conditional branches identical | □ |
```

#### 4. Model Configuration Parameters

**⚠️ CRITICAL: Never guess default parameters - always verify with user!**

When the original model uses `from_pretrained()` (diffusers `ModelMixin`) or reads from `config.json`, the model parameters are determined by the checkpoint's config file, NOT hardcoded defaults.

| Wrong Approach | Correct Approach |
|----------------|------------------|
| ❌ Copy default values from original code | ✅ Ask user for actual config from checkpoint |
| ❌ Use generic defaults (dim=2048, num_heads=16) | ✅ Get dim, ffn_dim, num_heads, num_layers from config.json |
| ❌ Assume parameters work with weights | ✅ Verify parameters match weight tensor shapes |

**Example: LiveAct config.json shows:**
```json
{
  "dim": 5120,
  "ffn_dim": 13824,
  "num_heads": 40,
  "num_layers": 40,
  "in_dim": 36
}
```

If using wrong defaults (dim=2048, num_heads=16), weight loading will fail with shape mismatch errors.

**Required Action:**
When implementing model `__init__`, **AskUserQuestion** to request:
1. `config.json` content from checkpoint directory
2. Or key parameters: `dim`, `ffn_dim`, `num_heads`, `num_layers`, `in_dim`, etc.
3. Update default parameter values to match actual checkpoint config

#### 5. Correct vs Incorrect Example

```python
# ❌ WRONG - Unnecessary "optimization"
def forward(self, x, grid_sizes, freqs):
    q = causal_rope_apply(x.flatten(2), grid_sizes, freqs)  # Wrongly added flatten
    q = q.view(B, -1, self.num_heads, self.head_dim)

# ✅ CORRECT - Faithful copy
def forward(self, x, grid_sizes, freqs):
    q = causal_rope_apply(x, grid_sizes, freqs)  # Identical to original
    q = q.transpose(1, 2)
```

### DiT Model Implementation

```python
# telefuser/models/<model>_dit.py
from telefuser.core.base_model import BaseModel

class MyModelDiT(BaseModel):
    def __init__(self, config: MyModelDiTConfig):
        super().__init__()
        # Directly copy from original - DO NOT modify
        self.x_embedder = nn.Linear(...)
        self.transformer_blocks = nn.ModuleList([...])

    def forward(self, hidden_states, timestep, encoder_hidden_states, ...):
        # Directly copy from original - DO NOT modify logic
        ...

    def get_fsdp_module_names(self) -> list[str]:
        return ["TransformerBlock", "SingleTransformerBlock"]

    @staticmethod
    def state_dict_converter():
        return MyModelDiTStateDictConverter()


class MyModelDiTStateDictConverter:
    def from_diffusers(self, state_dict: dict) -> dict:
        return state_dict  # or key remapping

    def from_official(self, state_dict: dict) -> dict:
        # Key remapping from official/BFL format
        ...
```

### Loading in Pipeline

```python
# DiT - source-level
transformer = Flux2DiT.from_pretrained(transformer_path, torch_dtype)
mm.add_module(transformer, "transformer")

# VAE/TextEncoder - HuggingFace loading
mm.load_from_huggingface(vae_path, module_source="diffusers",
                         module_class=AutoencoderKLFlux2, module_name="vae")
mm.load_from_huggingface(text_encoder_path, module_source="transformers",
                         module_class=Qwen3ForCausalLM, module_name="text_encoder")
```

### 🛑 Phase 3 Checkpoint

After completion:
1. Show `<model>_dit.py` implementation
2. Show state_dict_converter
3. **Run diff comparison and show results**
4. **AskUserQuestion**: "Phase 3 complete. Ready for Phase 4?"

---

## Phase 4: Code Cleanup

### Remove
- `gradient_checkpointing` attributes
- `self.training` conditionals
- Duplicate definitions (RMSNorm, swish, etc.)
- Unused code

### Standardize
- Consistent `from_pretrained` parameter names
- Encoders return dataclass (not dict)
- Shared utilities in single location

### 🛑 Phase 4 Checkpoint

After completion:
1. Run `pre-commit run --all-files`
2. Show cleanup summary
3. Update PROGRESS.md status
4. **AskUserQuestion**: "Phase 4 complete. Ready for Phase 5?"

---

## Phase 5: Review & Compare

### Goals
1. Compare pipeline logic with original
2. Verify edge case handling
3. Ensure numerical output matches

### Comparison Checklist

| Aspect | Check |
|--------|-------|
| Pipeline flow | Steps match original? |
| Edge cases | CFG=1, batch>1, custom sizes? |
| Model config | Parameters match? |
| Numerical | Output matches original? |

Create `examples/<model_name>/COMPARISON_REPORT.md` with findings.

### 🛑 Phase 5 Checkpoint

After completion:
1. Show comparison summary
2. Highlight any mismatches
3. If issues: provide fix suggestions
4. **AskUserQuestion**: "Integration complete. What next?"

---

## Skip Handling

When user says "skip X":
- Skip the work, NOT the checkpoint
- Still analyze from code if skipping execution
- Still use AskUserQuestion for approval

---

## Context Management

Phase 2.2 and 3 are critical and need precision. If context is exhausted after earlier phases, recommend starting a fresh session.

---

## Related Documentation

| Topic | Document |
|-------|----------|
| Model Implementation | `docs/en/adding_new_model.md` |
| Stage Implementation | `docs/en/adding_new_stage.md` |
| Attention Config | `docs/en/attention.md` |
| Parallel Inference | `docs/en/parallel.md` |
| Optimization | Use `/optimize-pipeline` skill after integration |