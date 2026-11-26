#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render sanitized Ascend NPU performance boards for the repository and bilingual MkDocs site.

Invariants enforced on every input JSON and rendered output:
1. Zero host/cluster/IP/user/path leakage: rejects any forbidden token immediately.
2. Explicit chip SKU and card count: every entry must declare an exact Ascend 910B2 or 910B4
   SKU with memory size (for example, 'Ascend 910B2 (64GB)' or 'Ascend 910B4 (32GB)') and an
   integer card_count >= 1, rendered as '<N>x <SKU>'.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

VALID_CHIP_PATTERN = re.compile(r"^Ascend\s+910B[24](?:C)?\s*\(\d+GB\)$")

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("server_node_name", re.compile(r"\b(?:node|gpu|npu|cpu|host)[-_]?\d+\b", re.IGNORECASE)),
    ("baremetal_host_name", re.compile(r"\bpm-[a-z0-9]+\b", re.IGNORECASE)),
    (
        "private_ipv4",
        re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    ),
    ("user_at_host", re.compile(r"\b[a-z_][a-z0-9_-]*@[a-z0-9._-]+\b", re.IGNORECASE)),
    ("internal_mount_path", re.compile(r"(?:/home/[a-z0-9_-]+|/mnt/[a-z0-9_-]+|/data[a-z0-9_-]*/)", re.IGNORECASE)),
)


def assert_sanitized_text(text: str, *, context: str) -> None:
    """Raise ValueError if text contains any server hostname, IP, username, or internal path."""
    for rule_name, pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ValueError(f"Sanitization violation [{rule_name}] in {context}: matched {match.group(0)!r}")


def validate_hardware_fields(entry: dict[str, Any], *, source: str) -> None:
    """Validate that chip_model explicitly names 910B2 or 910B4 and card_count is a positive int."""
    chip = entry.get("chip_model", "")
    if not isinstance(chip, str) or not VALID_CHIP_PATTERN.match(chip):
        raise ValueError(
            f"{source}: chip_model must explicitly state 910B2 or 910B4 with memory "
            f"(e.g. 'Ascend 910B2 (64GB)' or 'Ascend 910B4 (32GB)'), got {chip!r}"
        )
    cards = entry.get("card_count")
    if not isinstance(cards, int) or isinstance(cards, bool) or cards < 1:
        raise ValueError(f"{source}: card_count must be an integer >= 1, got {cards!r}")
    if not entry.get("parallel_spec"):
        raise ValueError(f"{source}: parallel_spec is required")
    if entry.get("cpu_arch") not in {"x86_64", "aarch64"}:
        raise ValueError(f"{source}: cpu_arch must be 'x86_64' or 'aarch64', got {entry.get('cpu_arch')!r}")


def load_records_and_slots(
    records_dir: Path,
    slots_path: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate verified benchmark records and remaining pending slots."""
    records: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        assert_sanitized_text(raw, context=path.name)
        data = json.loads(raw)
        validate_hardware_fields(data, source=path.name)
        records.append(data)

    covered = {
        (r["model"], r["task"], r["chip_model"], r["card_count"], r["cpu_arch"], r["cann_version"]) for r in records
    }

    pending_slots: list[dict[str, Any]] = []
    if slots_path and slots_path.is_file():
        raw_slots = slots_path.read_text(encoding="utf-8")
        assert_sanitized_text(raw_slots, context=slots_path.name)
        for slot in json.loads(raw_slots):
            validate_hardware_fields(slot, source=slot.get("slot_id", slots_path.name))
            key = (
                slot["model"],
                slot["task"],
                slot["chip_model"],
                slot["card_count"],
                slot["cpu_arch"],
                slot["cann_version"],
            )
            if key not in covered:
                pending_slots.append(slot)

    records.sort(key=lambda r: (r["model"], r["cpu_arch"], r["cann_version"], r["card_count"]))
    return records, pending_slots


def format_speedup(tf_warm: float | None, omni_warm: float | None) -> str:
    """Format speedup ratio and latency delta percentage against vLLM-Omni."""
    if not tf_warm or not omni_warm:
        return "—"
    ratio = omni_warm / tf_warm
    delta_pct = (tf_warm - omni_warm) / omni_warm * 100.0
    sign = "+" if delta_pct > 0 else ""
    return f"{ratio:.2f}x ({sign}{delta_pct:.1f}%)"


def _verified_table_rows(records: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for rec in records:
        wl = rec["workload"]
        tf = rec["telefuser"]
        om = rec.get("vllm_omni") or {}
        qg = rec.get("quality_gate") or {}
        hw_label = f"**{rec['card_count']}× {rec['chip_model']}**"
        stack = f"`{rec['cpu_arch']}` · CANN `{rec['cann_version']}`"
        workload_str = f"{wl['resolution']}, {wl['num_frames']}f, {wl['steps']} steps"
        tf_cell = f"**{tf['warm_median_s']:.2f} s** ({tf.get('denoise_s', 0):.1f} s)"
        om_warm = om.get("warm_median_s")
        om_cell = (
            f"{om_warm:.2f} s ({om.get('denoise_s', 0):.1f} s) `{om.get('version_or_commit', '')}`" if om_warm else "—"
        )
        speedup = format_speedup(tf.get("warm_median_s"), om_warm)
        if tf.get("peak_hbm_mb") and om.get("peak_hbm_mb"):
            hbm = f"{tf['peak_hbm_mb']:,} MB (vs. {om['peak_hbm_mb']:,} MB)"
        elif tf.get("peak_hbm_mb"):
            hbm = f"{tf['peak_hbm_mb']:,} MB"
        else:
            hbm = "—"
        q_items: list[str] = []
        if qg.get("psnr_db") is not None:
            q_items.append(f"PSNR {qg['psnr_db']:.2f} dB")
        if qg.get("audio_cosine") is not None:
            q_items.append(f"AudioCos {qg['audio_cosine']:.4f}")
        if not qg.get("nan_or_inf", True):
            q_items.append("Finite")
        rows.append(
            f"| `{rec['model']}` ({rec['task']}) | {hw_label} | {rec['parallel_spec']} | {stack} | "
            f"{workload_str} | {tf_cell} | {om_cell} | **{speedup}** | {hbm} | {', '.join(q_items)} |"
        )
    return rows


def _pending_table_rows(pending_slots: list[dict[str, Any]], *, status_label: str) -> list[str]:
    rows: list[str] = []
    for slot in pending_slots:
        hw_label = f"**{slot['card_count']}× {slot['chip_model']}**"
        stack = f"`{slot['cpu_arch']}` · CANN `{slot['cann_version']}` (`{slot['base_image']}`)"
        rows.append(
            f"| `{slot['model']}` ({slot['task']}) | {hw_label} | {slot['parallel_spec']} | {stack} | "
            f"{slot['workload_summary']} | `{status_label}` |"
        )
    return rows


def render_docs_en(records: list[dict[str, Any]], pending_slots: list[dict[str, Any]]) -> str:
    """Render docs/en/benchmark_ascend_npu.md."""
    header_verified = (
        "| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN | Workload "
        "| TeleFuser Warm (DiT) | vLLM-Omni Warm (DiT) | Speedup vs. vLLM-Omni | Peak HBM / Card | Quality Gate |"
    )
    header_pending = (
        "| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN / Base Image | Target Workload | Status |"
    )
    lines = [
        "# Ascend NPU Benchmarks",
        "",
        "This page tracks apples-to-apples performance and numerical parity between **TeleFuser** and **vLLM-Omni** on",
        "Huawei Ascend NPUs. Every entry explicitly records the **accelerator SKU** (`Ascend 910B2 (64GB)` vs.",
        "`Ascend 910B4 (32GB)`), **card count**, **parallel topology**, **CPU architecture**, and **CANN runtime**.",
        "",
        "See [Run Wan2.2 on Ascend 910B](ascend_910b.md) for container setup and",
        "[Hardware Platforms](platforms.md) for dispatch boundaries.",
        "",
        "## Verified Head-to-Head Results",
        "",
        "Warm latency reports the median of three consecutive warm requests after one cold warmup run on an identical",
        "software stack and checkpoint revision (values in parentheses show the denoising loop duration).",
        "",
        header_verified,
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
        *_verified_table_rows(records),
    ]
    if pending_slots:
        lines.extend(
            [
                "",
                "## Qualification Roadmap Matrix",
                "",
                header_pending,
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
                *_pending_table_rows(pending_slots, status_label="PENDING"),
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducible Five-Gate Harness",
            "",
            "All rows above are produced by the automated harness in",
            "`benchmarks/ascend_npu/` (`python benchmarks/ascend_npu/render_board.py`):",
            "",
            "1. **Gate G0 (Device & HCCL Probe)**: Verifies clean device visibility, NPU fused ops, HCCL",
            "   `all_reduce` / `all_to_all_single` bandwidth, and host-staged worker tensor queues.",
            "2. **Gate G1 (Single-Card Parity)**: Verifies finite pre-encode outputs, `ffprobe` media stream contracts",
            "   (including 32 kHz stereo audio sync for audio-video models), and `PSNR` / `MAE` parity.",
            "3. **Gate G2 (Multi-Card Scaling & Isolation)**: Verifies multi-NPU equivalence against single-NPU",
            "   outputs and concurrent multi-instance HCCL 16-port lease isolation.",
            "4. **Gate G3 (Warm Latency & Memory Profile)**: Measures `1 Cold + 3 Warm` serial requests and per-card",
            "   HBM peaks on the shared `quay.io/ascend/vllm-omni` container base.",
            "5. **Gate G4 (Clean-Room Reproduction)**: Executes a standalone `REPRODUCE.sh` with a stripped",
            "   environment (`env -i`) and zero source-tree bind mounts to guarantee third-party reproducibility.",
            "",
        ]
    )
    doc = "\n".join(lines)
    assert_sanitized_text(doc, context="docs/en/benchmark_ascend_npu.md")
    return doc


def render_docs_zh(records: list[dict[str, Any]], pending_slots: list[dict[str, Any]]) -> str:
    """Render docs/zh/benchmark_ascend_npu.md."""
    header_verified = (
        "| 模型 / 任务 | 芯片规格与卡数 | 并行策略 | 架构 / CANN | 工作负载 "
        "| TeleFuser 热态 (DiT) | vLLM-Omni 热态 (DiT) | 相对 vLLM-Omni | 单卡 HBM 峰值 | 质量门禁 |"
    )
    lines = [
        "# 昇腾 NPU 性能基准",
        "",
        "本页持续跟踪 **TeleFuser** 与 **vLLM-Omni** 在华为昇腾 NPU 上的同底座端到端性能与数值精度对齐结果。",
        "每一条记录均显式标明 **芯片具体型号**（区分 `Ascend 910B2 (64GB)` 与 `Ascend 910B4 (32GB)`）、**推理卡数**、",
        "**并行拓扑**、**CPU 架构**与 **CANN 版本**。",
        "",
        "容器构建与运行方式参见[《昇腾 910B 镜像》](ascend_910b.md)，",
        "跨硬件算子分派边界参见[《硬件平台》](platforms.md)。",
        "",
        "## 已验证同底座对标结果",
        "",
        "热态延迟取同一容器底座与权重版本下「1 次冷启动预热 + 连续 3 次热请求」的中位数（括号内为 DiT 去噪阶段耗时）。",
        "",
        header_verified,
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
        *_verified_table_rows(records),
    ]
    if pending_slots:
        lines.extend(
            [
                "",
                "## 待回填验收矩阵",
                "",
                "| 模型 / 任务 | 芯片规格与卡数 | 并行策略 | 架构 / CANN / 底座镜像 | 标准工作负载 | 状态 |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
                *_pending_table_rows(pending_slots, status_label="PENDING"),
            ]
        )
    lines.extend(
        [
            "",
            "## 五阶可复现验收门禁",
            "",
            "上表数据统一由 `benchmarks/ascend_npu/` 下的自动化验收脚手架产出，并通过",
            "`python benchmarks/ascend_npu/render_board.py` 自动校验脱敏与硬件规格字段后生成，禁止手工填数：",
            "",
            "1. **G0（设备与通信微基准）**：核验逻辑卡映射、NPU 融合算子、"
            "HCCL `all_reduce` / `all_to_all_single` 带宽与主存张量队列。",
            "2. **G1（单卡功能与真值对齐）**：核验编码前张量无 `NaN/Inf`、`ffprobe` 音视频流契约"
            "（含音视频模型 32 kHz 立体声同步）及 `PSNR` / `MAE`。",
            "3. **G2（多卡扩展与多实例隔离）**：核验多卡并行相对单卡同 seed 的输出一致性，"
            "以及多实例并发 HCCL 16 端口锁隔离。",
            "4. **G3（热态延迟与显存剖析）**：在同一 `quay.io/ascend/vllm-omni` 容器底座上执行 "
            "`1 Cold + 3 Warm` 计时与各卡 HBM 峰值采样。",
            "5. **G4（无状态盲跑门禁）**：通过 `env -i` 清空自定义环境变量、不挂载宿主机源码目录，"
            "独立执行生成的 `REPRODUCE.sh` 验证第三方一次跑通。",
            "",
        ]
    )
    doc = "\n".join(lines)
    assert_sanitized_text(doc, context="docs/zh/benchmark_ascend_npu.md")
    return doc


def render_repo_readme(records: list[dict[str, Any]], pending_slots: list[dict[str, Any]]) -> str:
    """Render benchmarks/ascend_npu/README.md."""
    header_verified = (
        "| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN | Workload "
        "| TeleFuser Warm (DiT) | vLLM-Omni Warm (DiT) | Speedup vs. vLLM-Omni | Peak HBM / Card | Quality Gate |"
    )
    header_pending = (
        "| Model / Task | Accelerator SKU & Cards | Parallelism | Arch / CANN / Base Image | Target Workload | Status |"
    )
    lines = [
        "# Ascend NPU Benchmark & Qualification Suite",
        "",
        "Reproducible qualification gates (`G0`–`G4`) and sanitized head-to-head performance records comparing",
        "**TeleFuser** with **vLLM-Omni** on Ascend 910B2 / 910B4 NPUs.",
        "",
        "## Verified Results",
        "",
        header_verified,
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
        *_verified_table_rows(records),
    ]
    if pending_slots:
        lines.extend(
            [
                "",
                "## Planned Qualification Matrix",
                "",
                header_pending,
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
                *_pending_table_rows(pending_slots, status_label="PENDING"),
            ]
        )
    lines.extend(
        [
            "",
            "## Updating the Board",
            "",
            "1. Run the qualification harness (`harness/preflight_check.sh`, `harness/comm_probe.py`,",
            "   `harness/bench_vs_vllm_omni.py`, and `harness/cleanroom_repro.sh`) on a clean container built from",
            "   `docker/Dockerfile.ascend`.",
            "2. Place the resulting sanitized JSON record into `benchmarks/ascend_npu/records/`.",
            "3. Regenerate the repository README and bilingual documentation pages:",
            "",
            "```bash",
            "python benchmarks/ascend_npu/render_board.py",
            "pytest tests/unit/benchmarks/test_ascend_npu_board.py",
            "```",
            "",
            "The renderer automatically rejects any record containing server hostnames, internal IP addresses,",
            "usernames, or internal mount paths, and requires explicit `Ascend 910B2 (64GB)` / `Ascend 910B4 (32GB)`",
            "SKU and card-count fields.",
            "",
        ]
    )
    doc = "\n".join(lines)
    assert_sanitized_text(doc, context="benchmarks/ascend_npu/README.md")
    return doc


def render_all(repo_root: Path) -> tuple[str, str, str]:
    """Load records from repo_root and return (readme_md, docs_en_md, docs_zh_md)."""
    bench_dir = repo_root / "benchmarks" / "ascend_npu"
    records, slots = load_records_and_slots(bench_dir / "records", bench_dir / "planned_slots.json")
    return (
        render_repo_readme(records, slots),
        render_docs_en(records, slots),
        render_docs_zh(records, slots),
    )


def main() -> None:
    """CLI entrypoint to regenerate README and bilingual MkDocs benchmark pages."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    args = parser.parse_args()

    readme_md, docs_en_md, docs_zh_md = render_all(args.repo_root)
    readme_out = args.repo_root / "benchmarks" / "ascend_npu" / "README.md"
    en_out = args.repo_root / "docs" / "en" / "benchmark_ascend_npu.md"
    zh_out = args.repo_root / "docs" / "zh" / "benchmark_ascend_npu.md"

    readme_out.write_text(readme_md, encoding="utf-8")
    en_out.write_text(docs_en_md, encoding="utf-8")
    zh_out.write_text(docs_zh_md, encoding="utf-8")
    print(f"Updated:\n  {readme_out}\n  {en_out}\n  {zh_out}")


if __name__ == "__main__":
    main()
