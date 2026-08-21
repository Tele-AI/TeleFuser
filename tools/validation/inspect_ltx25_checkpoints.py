"""Write a metadata-only provenance manifest for an LTX-2.5 distilled model pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from telefuser.models.ltx25.checkpoint import inspect_model_pack


def main() -> None:
    """Inspect required LTX-2.5 split checkpoints without materializing tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash checkpoint payloads; this reads every file in full.",
    )
    args = parser.parse_args()

    components = inspect_model_pack(args.model_root, include_sha256=args.sha256)
    manifest: dict[str, object] = {
        "model_root": str(args.model_root.resolve()),
        "sha256_included": args.sha256,
        "components": {name: metadata.as_dict() for name, metadata in components.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "components": sorted(manifest["components"])}, sort_keys=True))


if __name__ == "__main__":
    main()
