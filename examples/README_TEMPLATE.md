<!--
Copy this file to examples/{example_directory}/README.md and replace every
{PLACEHOLDER}. Delete guidance comments and optional sections that do not apply.

Keep the required section order so readers can scan all example READMEs in the
same way. Commands must be runnable from the repository root. Document only
options and behavior that exist in the checked-in example scripts.
-->

# {MODEL_FAMILY} Examples

{ONE_OR_TWO_SENTENCES_DESCRIBING_THE_SUPPORTED_MODELS_TASKS_AND_OUTPUTS}

## Model Source

<!-- Required. Add one row per checkpoint or auxiliary model. Use N/A when a source is unavailable. -->

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| `{MODEL_NAME}` | [{HF_REPOSITORY}]({HF_URL}) | [{MODELSCOPE_REPOSITORY}]({MODELSCOPE_URL}) | {MODEL_PURPOSE} |

## Feature Support

<!--
Required. Keep only relevant rows and add model/version columns when support
differs. Use Supported, Unsupported, Partial, or N/A; explain Partial below the
table. Do not use an ambiguous question mark for unverified support.
-->

| Feature | Support | Notes |
| --- | --- | --- |
| {TASK_OR_FEATURE} | Supported | {CONSTRAINTS_OR_VARIANTS} |
| Multi-GPU inference | {SUPPORT_STATUS} | {PARALLEL_STRATEGY_AND_VALID_DEGREES} |
| LoRA | {SUPPORT_STATUS} | {SUPPORTED_VARIANTS} |
| Quantization | {SUPPORT_STATUS} | {DTYPES_OR_FORMATS} |
| CPU offload | {SUPPORT_STATUS} | {OFFLOAD_MODES} |
| Feature cache | {SUPPORT_STATUS} | {CACHE_IMPLEMENTATION} |
| Server API | {SUPPORT_STATUS} | {SERVE_OR_STREAM_SERVE} |

## Requirements

<!-- Required. State minimum hardware and extra dependencies beyond the normal TeleFuser installation. -->

- GPU: {GPU_MODEL_OR_MINIMUM_VRAM}
- Software: {CUDA_PYTORCH_OR_EXTRA_PACKAGE_REQUIREMENTS}
- Input assets: {REQUIRED_INPUT_FORMATS_OR_NONE}

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup). Then install any
example-specific dependencies:

```bash
{INSTALL_COMMANDS_OR_COMMENT_STATING_NO_EXTRA_DEPENDENCIES}
```

## Model Directory

<!-- Required. Show the exact layout expected by the scripts. Omit branches that are downloaded automatically. -->

```text
{MODEL_ROOT}/
|-- {CHECKPOINT_OR_DIRECTORY}
\-- {AUXILIARY_CHECKPOINT_OR_DIRECTORY}
```

Set the model root if the examples use `TF_MODEL_ZOO_PATH`:

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Quick Start

<!-- Required. Lead with the smallest representative command that produces an output. -->

```bash
python examples/{example_directory}/{representative_script}.py \
  --model_root /path/to/model \
  --prompt "{EXAMPLE_PROMPT}" \
  --output_path work_dirs/{OUTPUT_FILE}
```

The command writes {OUTPUT_DESCRIPTION} to `work_dirs/{OUTPUT_FILE}`.

## Examples

<!--
Required. Group scripts by task when the directory contains multiple tasks.
Repeat the task and script blocks as needed. Use script names as headings.
-->

### {TASK_NAME}

#### `{script_name.py}`

{ONE_SENTENCE_PURPOSE_AND_WHEN_TO_USE_THIS_SCRIPT}

```bash
# Basic usage
python examples/{example_directory}/{script_name.py} \
  --model_root /path/to/model \
  {REQUIRED_ARGUMENTS}

# Multi-GPU or another important variant
python examples/{example_directory}/{script_name.py} \
  --gpu_num {GPU_COUNT} \
  --model_root /path/to/model \
  {VARIANT_ARGUMENTS}
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--model_root` | `{DEFAULT_OR_NONE}` | {MODEL_ROOT_DESCRIPTION} |
| `--gpu_num` | `{DEFAULT_GPU_COUNT}` | {GPU_COUNT_CONSTRAINTS} |
| `{OPTION}` | `{DEFAULT}` | {OPTION_DESCRIPTION} |

Key behavior:

- {IMPORTANT_DEFAULT_OR_MODEL_VARIANT}
- {OUTPUT_SHAPE_FORMAT_OR_LOCATION}
- {LIMITATION_OR_RESOURCE_NOTE}

## Configuration

<!--
Optional. Keep this section only when users must understand non-obvious
parallel, cache, scheduler, or runtime rules.
-->

### {CONFIGURATION_TOPIC}

{EXPLAIN_THE_RULE_ITS_DEFAULT_AND_WHEN_TO_CHANGE_IT}

```python
{MINIMAL_CONFIGURATION_SNIPPET}
```

## Serving

<!-- Optional. Keep only when at least one script implements a serve or stream-serve contract. -->

Start the service:

```bash
telefuser {serve_or_stream-serve} examples/{example_directory}/{server_script}.py \
  --port {PORT} \
  {OTHER_REQUIRED_OPTIONS}
```

See the [service guide](../../docs/en/service.md) or
[stream server guide](../../docs/en/stream_server.md) for API and deployment details.

## Performance

<!--
Optional. Include only measured results. Never commit TBD values. State enough
environment and workload detail for another developer to reproduce the result.
Use the same metric definition for every row.
-->

Measured with {GPU_COUNT_AND_MODEL}, {SOFTWARE_VERSIONS}, and commit `{GIT_REVISION}`. Results use {PRECISION},
{ATTENTION_BACKEND}, and exclude {EXCLUDED_PHASES_OR_NOTHING}.

| Configuration | GPUs | Resolution | Frames | Steps | Time (s) | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| {CONFIGURATION_NAME} | {GPU_COUNT} | {RESOLUTION} | {FRAME_COUNT} | {STEP_COUNT} | {ELAPSED_TIME} | {PEAK_VRAM} |

Reproduce the measurement:

```bash
{BENCHMARK_COMMAND}
```

## Troubleshooting

<!-- Optional. Include only failures specific to this example; link shared issues to the relevant guide. -->

### {ERROR_OR_SYMPTOM}

{CAUSE_AND_ACTIONABLE_FIX}

```bash
{DIAGNOSTIC_OR_FIX_COMMAND}
```

## Notes

<!-- Optional. Keep model-specific limitations or output semantics that do not fit above. -->

- {MODEL_SPECIFIC_LIMITATION_OR_COMPATIBILITY_NOTE}
- {OUTPUT_OR_QUALITY_NOTE}
