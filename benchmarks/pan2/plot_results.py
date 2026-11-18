"""Create the PAN2 NVFP4 result chart from report.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).parent
run = json.loads((root / "report.json").read_text())["runs"][0]
metrics = {
    "Latency (s)": run["latency_seconds"],
    "Peak allocated (GiB)": run["peak_allocated_gib"],
    "Generated FPS": run["generated_video_fps"],
    "Realtime ratio": run["realtime_ratio"],
}
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.5))
for axis, (label, value) in zip(axes, metrics.items()):
    axis.bar(["TeleFuser\nNVFP4"], [value], color="#2563eb")
    axis.set_title(label)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.text(0, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
fig.suptitle("PAN2 layerwise NVFP4 on RTX 5090 | 256x256, 17 frames, 1 step")
fig.tight_layout()
fig.savefig(root / "pan2-nvfp4-results.png", dpi=180)
