# Supported Models

This catalog lists the maintained model-family guides published with TeleFuser. A model's Cookbook page is the
source of truth for checkpoint identifiers, Hugging Face and ModelScope locations, required files, supported tasks,
hardware assumptions, and runnable commands.

"Supported" means that TeleFuser contains a maintained pipeline or example contract. It does not mean that every
checkpoint, precision, attention backend, or GPU topology is interchangeable. Use the validated profile documented
by the selected guide before changing optimization settings.

## World Models and Real-Time

| Model | Tasks | Execution path |
| --- | --- | --- |
| [LingBot-World v2](/TeleFuser/cookbook/lingbot-world/) | Bidirectional world-model streaming | Standalone and LiveKit |
| [LingBot-World-Fast](/TeleFuser/cookbook/lingbot-world/) | Causal interactive streaming | Standalone and LiveKit |
| [ABot-World-0-5B-LF](/TeleFuser/cookbook/abot-world/) | Interactive generation | HTTP and LiveKit |

## Video, Audio, and Restoration

| Model | Tasks | Execution path |
| --- | --- | --- |
| [WanVideo](/TeleFuser/cookbook/wan-video/) | T2V, I2V, FL2V | Standalone and batch service |
| [LTX-2.3](/TeleFuser/cookbook/ltx23/) | I2V with audio | Standalone and batch service |
| [LTX-2.5 Distilled](/TeleFuser/cookbook/ltx25-distilled/) | T2V and I2V with audio | Standalone and multi-GPU |
| [MiniMax H3](/TeleFuser/cookbook/minimax-h3/) | T2VA, FL2VA, Ref2VA | Standalone and batch service |
| [LingBot-Video](/TeleFuser/cookbook/lingbot-video/) | T2I, T2V, TI2V, refinement | Standalone and multi-GPU |
| [LongCat-Video](/TeleFuser/cookbook/longcat-video/) | T2V, I2V, continuation | Standalone and batch service |
| [LiveAct](/TeleFuser/cookbook/liveact/) | Speech-to-video | Standalone and streaming |
| [FlashVSR](/TeleFuser/cookbook/flashvsr/) | Video super-resolution | Standalone and streaming |
| [SwiftVR](/TeleFuser/cookbook/swiftvr/) | Causal video restoration | Standalone and multi-GPU |

## Image Generation

| Model | Tasks | Execution path |
| --- | --- | --- |
| [Qwen-Image](/TeleFuser/cookbook/qwen-image/) | T2I and editing | Standalone and batch service |
| [Z-Image](/TeleFuser/cookbook/z-image/) | T2I | Standalone and batch service |
| [Flux2 Klein](/TeleFuser/cookbook/flux2-klein/) | T2I | Standalone and batch service |

## Vision-Language-Action

| Model | Tasks | Execution path |
| --- | --- | --- |
| [LingBot-VLA v2](/TeleFuser/cookbook/lingbot-vla-v2/) | Robot action prediction | Standalone inference |

## Selecting a Profile

1. Select a model family by task.
2. Open its Cookbook page and use one of the documented model sources.
3. Start from the validated hardware and precision profile.
4. Run the documented standalone command before enabling serving or distributed execution.
5. Change one optimization axis at a time and retain the guide's output validation.

For the smallest onboarding path, use the Wan2.1 1.3B profile in [Basic Inference](quickstart.md).
