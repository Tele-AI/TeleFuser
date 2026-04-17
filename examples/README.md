# TeleFuser Example Runner

Runs configured pipelines in isolated subprocesses, compares outputs against
baselines (PSNR/SSIM for video, pixel diff for image), and prints a results table.

## Quick Start

```bash
# List configured pipelines
python examples/run_examples.py --list

# Run a specific pipeline
python examples/run_examples.py --pipeline wan21_1_3b_t2v

# Run all enabled pipelines
python examples/run_examples.py --all

# Update baselines after successful runs
python examples/run_examples.py --all --update-baseline

# Resume from previous run (skip pipelines with existing outputs)
python examples/run_examples.py --all -r work_dirs/example_outputs/main_abc123_2026-04-17_1200

# Parallel execution across 4 GPUs
python examples/run_examples.py --all --gpus 0,1,2,3

# Resume with parallel execution
python examples/run_examples.py --all --gpus 0,1,2,3 -r work_dirs/example_outputs/main_abc123_2026-04-17_1200
```

## CLI Reference

```
python examples/run_examples.py [OPTIONS]

Options:
  --list                 List configured pipelines and exit
  --pipeline NAME        Run a specific pipeline by name
  --all                  Run all enabled pipelines
  --update-baseline      Update baseline outputs after successful runs
  --config PATH          Path to config YAML (default: example_config.yaml)
  -r, --resume DIR       Resume from existing output directory (skip completed pipelines)
  --gpus IDS             Parallel execution with specified GPUs (e.g., '0,1,2,3')
  -v, --verbose          Show real-time log output from each pipeline
```

## File Structure

```
examples/
  run_examples.py          # Single script: config, execution, metrics, reporting
  example_config.yaml      # Pipeline registry + configuration
  README.md                # This file
```

## Configuration

Edit `example_config.yaml` to manage pipelines:

```yaml
defaults:
  seed: 42
  timeout_seconds: 1800
  psnr_min: 25.0
  ssim_min: 0.85
  pixel_diff_max: 0.02

output_root: work_dirs/example_outputs

pipelines:
  wan21_1_3b_t2v:
    script: wan_video/wan21_1_3b_text_to_video_h100.py
    gpu_count: 1
    output_type: video
    model_root: /path/to/model
    ppl_config_overrides:
      attn_impl: FLASH_ATTN_2
```

### Pipeline Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| script | str | required | Path to example script (relative to `examples/`) |
| enabled | bool | true | Skip if false |
| gpu_count | int | 1 | GPUs to allocate |
| output_type | str | video | `video` or `image` |
| timeout_seconds | int | 1800 | Max execution time |
| seed | int | 42 | Random seed |
| model_root | str\|null | null | Override model directory |
| prompt | str\|null | null | Override generation prompt |
| input_image_path | str\|null | null | Input image for I2V / edit pipelines |
| input_video_path | str\|null | null | Input video for VSR / continue pipelines |
| ppl_config_overrides | dict | {} | Override PPL_CONFIG keys |
| psnr_min | float | 25.0 | Video: minimum PSNR vs baseline |
| ssim_min | float | 0.85 | Video: minimum SSIM vs baseline |
| pixel_diff_max | float | 0.02 | Image: max mean pixel difference |
| max_elapsed_seconds | float\|null | null | Performance threshold |
| max_gpu_memory_mb | float\|null | null | GPU memory threshold |

## Output

```
work_dirs/example_outputs/
  2026-04-02/                                  # Date-based output directory
    wan_video__wan21_1_3b_t2v_1gpu_480x832.mp4
    qwen_image__qwen_t2i_1gpu_1024x1024.png
    ...
  baseline/                                    # Baseline outputs (independent)
    wan_video__wan21_1_3b_t2v_1gpu_480x832.mp4
    ...
  logs/                                        # Log files (timestamped)
    20260402_120000_wan_video__wan21_1_3b_t2v_1gpu.log
    20260402_130000_qwen_image__qwen_t2i_1gpu.log
    ...
  example_report.json                          # JSON report with metrics + environment
```

### Output Naming Convention

**Output files:**
```
{example_dir}__{example_name}_{gpu_count}gpu_{resolution}.{ext}
```
Example: `wan_video__wan21_1_3b_text_to_video_h100_1gpu_480x832.mp4`

**Log files:**
```
{timestamp}_{example_dir}__{example_name}_{gpu_count}gpu.log
```
Example: `20260402_120000_wan_video__wan21_1_3b_text_to_video_h100_1gpu.log`

### Report Enhancement

The `example_report.json` now includes:
- **failed_details**: List of failed cases with error message, log path, reproduce command, and analysis hint
- **reproduce_all_failed**: Single command to reproduce all failed cases

## Resume Mode

When a test run is interrupted (e.g., OOM, timeout, manual stop), use `-r/--resume` to continue:

```bash
# Resume from previous output directory
python examples/run_examples.py --all -r work_dirs/example_outputs/main_abc123_2026-04-17_1200
```

**How it works:**
1. Checks for existing output files matching each pipeline's naming pattern
2. Skips pipelines that already have output files
3. Runs only the remaining (incomplete) pipelines
4. Merges results from completed pipelines with new results
5. Shows complete summary table including all pipelines
6. Saves combined report to `example_report.json`

**Result aggregation:**
- The final results table shows status for **all** configured pipelines
- Failed pipelines from previous run are included in summary counts
- Exit code reflects the combined status of all pipelines

## Parallel Execution

Use `--gpus` to run multiple pipelines concurrently:

```bash
# Run pipelines in parallel across 4 GPUs
python examples/run_examples.py --all --gpus 0,1,2,3
```

**Scheduling strategy:**
- Greedy allocation: pipelines sorted by GPU requirement (largest first)
- Each pipeline gets exclusive access to its allocated GPUs
- GPUs released back to pool when pipeline completes
- Automatic timeout handling with graceful shutdown

**Example scenario (4 GPUs):**
- Pipeline A (4 GPUs): runs exclusively on all GPUs
- Pipeline B (2 GPUs) + Pipeline C (2 GPUs): run concurrently after A completes
- Pipeline D (1 GPU): fills remaining slots while B/C run

## Features

- **Explicit registry**: `example_config.yaml` lists all pipelines — no auto-discovery
- **Subprocess isolation**: Each pipeline runs in its own process with pinned `CUDA_VISIBLE_DEVICES`
- **Baseline management**: First run auto-saves as baseline; `--update-baseline` refreshes
- **Regression metrics**: PSNR + SSIM for video, pixel diff for image
- **GPU memory tracking**: Peak VRAM usage per pipeline
- **Output validation**: NaN/Inf detection
- **Error classification**: MODEL_LOAD_ERROR, INFERENCE_ERROR, OUTPUT_ERROR, OOM_ERROR, TIMEOUT
- **Enhanced reporting**: Failed cases include reproduce commands and analysis hints
- **Resume support**: Skip completed pipelines when resuming from existing output directory
- **Parallel execution**: Run multiple pipelines concurrently across a pool of GPUs
- **Complete result summary**: Resume mode shows aggregated results for all pipelines (completed + new)