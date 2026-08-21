---
title: "Agent Dev First Through the MiniMax H3 Integration"
description: What two independent Codex Goals reveal about using architecture, contracts, validation, and durable artifacts to guide agents.
date: 2026-08-20
status: validated
validated_revision: 819c238
hardware: 1, 2, and 4 x NVIDIA H100 80 GB HBM3
tags:
  - agent-development
  - architecture
  - model-integration
  - validation
  - minimax-h3
---

# Agent Dev First Through the MiniMax H3 Integration

TeleFuser recently added support for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). The implementation
covers joint text-to-video-and-audio (T2VA), first/last-frame-to-video-and-audio (FL2VA), and
reference-to-video-and-audio (Ref2VA) generation. It supports synchronized audio-video output, stage-level CPU
offload on one GPU, resident execution on two or four GPUs, FSDP, Ulysses Sequence Parallelism, Tensor Parallelism,
and standard example and service entrypoints.

The integration was completed primarily through two independent Codex Goal sessions. According to the project
process record, no code was edited directly by a person; human involvement concentrated on setting the goals,
preparing the environment, and accepting final audio-video quality. This is not evidence that agents can complete
every software project without human engineering. It motivates a narrower question:

> Once an agent can search a codebase, infer a path from nearby implementations, and react to test feedback, how
> should the software system itself provide direction?

This article calls that design perspective **Agent Dev First (ADF)**. Instead of prescribing every action for one
task, ADF makes the repository's knowledge structure, module boundaries, validation paths, and durable artifacts
answer two questions: where should this change go, and how can the agent know it is correct?

## Validation Snapshot

| Field | Value |
|---|---|
| Status | `validated` |
| Validation revision | [`819c238`](https://github.com/Tele-AI/TeleFuser/commit/819c2388d7bbdc259821be9c6180879643a0c347) |
| Integration and acceptance dates | 2026-08-04 through 2026-08-06 |
| GPU | 1, 2, and 4 x NVIDIA H100 80 GB HBM3; final parallel acceptance used 4 x H100 |
| Software | Python 3.11.13, PyTorch 2.11.0+cu128, CUDA 12.8, NCCL 2.28.9 |
| Reference | A pinned local SGLang MiniMax H3 implementation |
| Faithful-path workloads | 768p; 10-second T2VA, 8-second FL2VA, and 5-second Ref2VA; all at 50 steps |
| Output contract | 24 FPS video, 32 kHz stereo audio, synchronized MP4 |
| Four-GPU regression | 768p, 5-second, 50-step T2VA with resident Ulysses2 x TP2 |

The validation claims apply to these checkpoints, requests, software versions, and hardware. They are not quality or
performance guarantees for other checkpoints, topologies, or generated content. The two Goal boundaries and the
absence of direct human code edits come from the project process record. Commits establish the implementation,
verification results, and handoff artifacts, but do not independently establish editor identity.

## Why MiniMax H3 Is a Useful Case

MiniMax H3 is not a video model that only requires loading a DiT checkpoint. A complete request includes at least:

```text
Prompt and reference media
  -> Text Encoder
  -> Video / Audio Condition VAE
  -> joint packing of text, video, and audio tokens
  -> joint DiT denoising
  -> separate video and audio scheduler updates
  -> Video VAE and Audio VAE decoding
  -> audio-video synchronization and MP4 muxing
```

T2VA, FL2VA, and Ref2VA also have different conditioning paths, material rules, time semantics, and geometry
sources. The port had to address official checkpoint conversion, FP32/BF16 precision boundaries, packed sequences,
multi-GPU parallelism, resident models, cross-stage transport, and service requests together.

The agent therefore faced a set of related engineering problems that still needed independent validation, not a
local "add one model class" task. That makes the integration a useful test of whether a repository can provide its
own direction.

## From Procedure Manuals to Engineering Structure

A common way to make an agent reliable is to write a detailed Skill: read these files first, edit them in this order,
handle each error this way, and run these checks at the end. Such procedure manuals remain useful for infrequent,
specialized, or cross-tool workflows.

As agents improve at code search, analogical implementation, and test-driven correction, they may need less
continuous step-by-step instruction. That does not make engineering design less important. With less procedural
guidance, the agent depends more heavily on the system to answer:

- Which module owns this feature?
- Which interface should express it?
- Which dependencies and changes are allowed?
- How can it obtain fast local feedback?
- What state counts as complete?
- How can a failure be located, handed off, and resumed?

Skills mainly answer *how to perform a procedure*. ADF emphasizes *where work belongs* and *how correctness is
established*. It neither turns every rule into a type check or CI job nor replaces Skills. It moves durable knowledge
into structure, interfaces, tests, and tools so that each new session does not have to remember it again.

## Two Goals: Correctness First, Execution Second

The two sessions shared no continuous conversation history. They handed work off through code, tests, frozen inputs,
manifests, report generators, and commits.

| Goal | What could change | What had to remain fixed | Key evidence |
|---|---|---|---|
| 1. Establish a faithful path | New model and pipeline code, checkpoint conversion, request planning, validation tools | Official task semantics, schedulers, precision boundaries, and output semantics | [`82de4e9`](https://github.com/Tele-AI/TeleFuser/commit/82de4e9bb128d170c0dd8e0b769a376c06957d3b), [`0db8807`](https://github.com/Tele-AI/TeleFuser/commit/0db8807e36d5cf278d36649d9c260617e05e230b), [`9b4dfb7`](https://github.com/Tele-AI/TeleFuser/commit/9b4dfb72f3ae5c78e3c7b3b0c109b647541a9377) |
| 2. Optimize execution | Model residency, SP/TP composition, stage transport, static layouts, buffers, and public ops/kernels | Goal 1's conditioning layout, scheduler semantics, precision boundaries, and output format | [`b3e1672`](https://github.com/Tele-AI/TeleFuser/commit/b3e1672cd0e3e0f14402a63548b6487858e08783), [`ecb261f`](https://github.com/Tele-AI/TeleFuser/commit/ecb261f80dde7a8062f9ee891c0b744798f6484f), [`b629b9e`](https://github.com/Tele-AI/TeleFuser/commit/b629b9e2343decfad2c6102124a5c9d651536445) |

The first Goal established model components, the three request planners, packed joint denoising, dual-modality
schedulers, complete audio-video output, and SGLang/TeleFuser trajectory and artifact comparison tools. Correctness
became the baseline only after the official 50-step T2VA, FL2VA, and Ref2VA reports passed their locked gates.

The second Goal added resident multi-GPU execution, Ulysses and TP composition, direct CUDA tensor transport between
stages, request-static layouts, denoising buffer reuse, and public ops and kernel optimizations. The later
[`a05d3ee`](https://github.com/Tele-AI/TeleFuser/commit/a05d3eec948930535a27c9a4a82225af36842d8d)
four-GPU regression and
[`a680135`](https://github.com/Tele-AI/TeleFuser/commit/a68013557cc37c7bc5b566e0f05a01aab7f4a6c6)
audio and service acceptance turned "the optimization preserves the output contract" into a repeatable check.

The commit timeline is an engineering evidence chain, not a transcript of the Goal sessions. The important property
is that the second session could recover task semantics and completion criteria from the system without reading the
first conversation.

## How TeleFuser Says Where Work Belongs

The integration did not use a step-by-step MiniMax H3-specific Skill. The agent inferred the implementation path from
the directory structure, interface contracts, neighboring implementations, and tests.

| Concern | Stable location | Primary boundary |
|---|---|---|
| Model mathematics and checkpoint conversion | `telefuser/models/` | Models use public `telefuser.ops`; they do not import internal kernels directly |
| Task semantics and material planning | `telefuser/pipelines/minimax_h3/` | T2VA, FL2VA, and Ref2VA differences remain in task profiles and request plans |
| Text Encoder, Denoising, Video/Audio VAE | Independent stages | Each stage owns its precision, offload, device, and lifecycle configuration |
| Parallel execution | Shared parallel config and distributed runtime | SP, TP, and FSDP reuse existing configuration rather than model-private framework APIs |
| Local and service execution | Standard entrypoints and manifests in `examples/minimax_h3/` | `PPL_CONFIG`, `get_pipeline`, `run`, and `run_with_file` share the service contract |
| Validation | `tests/`, `tools/validation/`, and frozen media | Fast local tests and expensive real generation form separate layers |

This structure did not tell the agent which line to edit first. It answered the more consequential question of where
model mathematics, task semantics, runtime parallelism, service contracts, and presentation belong.

Stable boundaries also constrained scope. MiniMax H3 introduced no new framework-level configuration fields,
environment variables, CLI system, or service schema. Required extensions stayed within the ownership of the model,
pipeline, examples, and existing public ops.

## ADF Also Needs a Definition of Done

Finding the correct file does not establish correctness. MiniMax H3 validation was divided into layers:

| Validation target | What it primarily establishes | Representative location |
|---|---|---|
| Checkpoint and numeric tests | Model structure, conversion, and FP32/BF16 boundaries | `tests/unit/models/test_minimax_h3_*.py` |
| Packed sequence, scheduler, and stage tests | Local computation and orchestration semantics | `tests/unit/pipelines/minimax_h3/` |
| Trajectory parity | Token layouts, initial noise, and scheduler boundaries match for fixed inputs | `compare_minimax_h3_trajectories.py` |
| Artifact parity | Video frames, waveforms, spectra, envelopes, lag, and container output pass gates | `compare_minimax_h3_artifacts.py` |
| Example and service parity | Service wrappers and direct entrypoints preserve one parameter contract | `test_examples.py`, `test_example_service_parity.py` |
| Four-GPU 50-step regression | Resident Ulysses2 x TP2 completes end-to-end audio-video generation | `minimax_h3_t2va_4gpu` registry |

Fast tests supply local feedback; real generation performs final acceptance. A failure can therefore be attributed to
model conversion, pipeline semantics, the parallel runtime, an external entrypoint, or the final artifact instead of
being compressed into one opaque service failure.

Frozen inputs are also part of the system interface. [`provenance.json`](https://github.com/Tele-AI/TeleFuser/blob/819c2388d7bbdc259821be9c6180879643a0c347/examples/data/minimax-h3/provenance.json)
records the official media sources, byte sizes, and SHA-256 hashes. The reference and candidate runners produce
manifests containing request, media, checkpoint, and artifact hashes. Large tensors and MP4 files can remain local
acceptance artifacts while source-controlled generators, hash rules, and comparators make them reconstructable.

## Reproduction Entrypoints

The CPU-side model, pipeline, contract, and comparator tests run independently:

```bash
pytest -q tests/unit/models/test_minimax_h3_*.py
pytest -q tests/unit/pipelines/minimax_h3
pytest -q tests/unit/service/test_example_service_parity.py
```

With local checkpoints and four H100 GPUs, create or check the standard regression:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
python examples/run_examples.py \
  --pipeline minimax_h3_t2va_4gpu \
  --gpus 0,1,2,3 \
  --update-baseline

TF_MODEL_ZOO_PATH=/path/to/model_zoo \
python examples/run_examples.py \
  --pipeline minimax_h3_t2va_4gpu \
  --gpus 0,1,2,3
```

The official-path comparison is composed from these tools:

```text
freeze_minimax_h3_reference.py
  -> run_minimax_h3_sglang_reference.py
  -> run_minimax_h3_telefuser_reference.py
  -> compare_minimax_h3_trajectories.py
  -> compare_minimax_h3_artifacts.py
```

They require a local MiniMax H3 checkpoint, the pinned SGLang reference checkout, and sufficient H100 resources.
Each script exposes its complete arguments through `--help`. Acceptance gates are written explicitly into JSON
reports rather than hidden in visual inspection. Final audio-video quality still receives human acceptance because
numeric similarity does not establish every aspect of conditioning fidelity and perceptual quality.

## Stronger Agents Expose Architecture Debt Faster

The integration also revealed three areas where TeleFuser can improve.

First, example regression, service parity, and related tests still contain duplicated registrations. Pipeline
manifests exist, but they are not yet the complete single source for the example registry, service contracts, and the
regression matrix.

Second, SP and TP were composable in code while documentation temporarily retained an obsolete mutual-exclusion
rule. Correct code does not cancel the search and reasoning cost of stale documentation.

Third, cross-process CUDA tensor ownership and shutdown order remain complex. Later commits hardened the lifecycle
with independent refcounters, channel-owned cleanup, and cooperative shutdown, but the stage DAG still cannot fully
express buffer, producer, consumer, and teardown dependencies.

Longer Skills cannot remove these problems at their source. A better direction is to let machine-readable pipeline
manifests drive entrypoint registration and regression matrices, and to make resource ownership explicit in the
stage DAG. Stronger agents do not erase architecture debt; they encounter it faster through executable
configuration, composition, and test feedback.

## Prompt, Skills, Harness, and ADF

This article uses the following division of responsibilities as an engineering model derived from this case. It is
not a normative definition of any particular agent product:

| Layer | Responsibility |
|---|---|
| Prompt | The current task's objective, constraints, and acceptance conditions |
| Skills | Procedures for specialized, infrequent, or cross-tool work |
| Harness | Agent scheduling, tool access, execution isolation, and feedback loops |
| ADF | Durable knowledge structure, module boundaries, handoff artifacts, and validation paths |
| Types, structural tests, and CI | Invariants that no participant may break |

When the same rule starts appearing in several Skills, the repository should ask whether every agent must continue
to read and remember it, or whether it belongs in architecture, interfaces, tests, and tools as part of the system.

## Related Work and Claim Boundary

[SWE-agent](https://arxiv.org/abs/2405.15793) treats the Agent-Computer Interface as an important design object for
software engineering agents. [SWE-bench](https://arxiv.org/abs/2310.06770) evaluates repository-level work with real
GitHub issues and tests. Repository instructions, Skills, executable specifications, architecture fitness
functions, and paved paths all reduce the cost of interacting with a software system.

ADF does not claim that these ideas are new, or that clear architecture benefits only agents. Its narrower claim is
that when an agent autonomously chooses local implementation steps, the repository must carry more responsibility
for navigation, constraints, validation, and cross-session handoff. MiniMax H3 provides one case backed by commits
and real-generation acceptance.

## Limitations and Next Steps

- This is one project, one model family, and one agent workflow; it does not represent every repository or agent.
- Two Goals show that this particular boundary could be handed off, not that fewer sessions are always better.
- Quality acceptance includes human judgment and is not yet a fully automated, publicly replayable perceptual eval.
- Large trajectory and MP4 artifacts are not stored in Git; the repository retains fixed inputs, hashes, generators,
  comparators, and acceptance summaries.
- ADF has not yet been evaluated in a controlled comparison against heavy Skills or other codebases.

Future comparisons could measure time to discover the correct entrypoint, time to first useful validation,
cross-module change size, handoff success rate, and recovery time after failure. The immediate TeleFuser work is to
make manifests drive examples, services, and regressions, and to make stage resource lifetimes explicit.

## Conclusion

MiniMax H3 was not integrated because the agent received a longer model-specific procedure manual. The work could
progress in stages because TeleFuser already offered stable locations for models, stages, pipelines, parallelism,
services, and validation.

The hardest problems appeared where acceptance conditions, duplicate registration, and resource ownership were not
yet expressed clearly by the system. This supports a limited but actionable conclusion: with explicit stage
boundaries, frozen acceptance artifacts, and contract tests, an agent can depend less on continuous conversation
memory and step-by-step instructions while completing correctness alignment and execution optimization in separate
phases.

Skills are procedure manuals for agents. Agent Dev First means shaping the engineering system so an agent can work in
it directly, validate locally, and hand work off durably. Procedures change with models and tools. Architecture
determines the cost at which every later agent can understand, modify, validate, and continue evolving the system.
