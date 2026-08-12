#!/usr/bin/env python3
"""Probe the SGLang-Omni Higgs import boundary.

This is intentionally narrower than a serving/runtime parity gate. The direct
Higgs modules are the source contract used by the one-step golden generator;
the full model import checks whether the local environment can even start the
SGLang-Omni runtime path.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


DIRECT_MODULES = (
    "sglang_omni.models.higgs_tts.text_tokenizer",
    "sglang_omni.models.higgs_tts.modeling",
    "sglang_omni.models.higgs_tts.hf_config",
)
FULL_MODEL_MODULE = "sglang_omni.models.higgs_tts.model"


def git_short_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def module_status(name: str) -> tuple[str, str]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        return "fail", f"{type(exc).__name__}:{exc}"
    return "ok", ""


def normalize_reason(reason: str) -> str:
    if "No module named 'sglang'" in reason or 'No module named "sglang"' in reason:
        return "missing_sglang"
    return reason.replace("\n", " ")


def print_kv(key: str, value: str) -> None:
    print(f"{key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sglang-omni-src",
        required=True,
        type=Path,
        help="SGLang-Omni source tree containing sglang_omni/.",
    )
    parser.add_argument(
        "--require-direct",
        action="store_true",
        help="Return nonzero if direct Higgs source modules cannot be imported.",
    )
    parser.add_argument(
        "--require-full-model",
        action="store_true",
        help="Return nonzero if the full SGLang-Omni Higgs model cannot be imported.",
    )
    args = parser.parse_args()

    src = args.sglang_omni_src.resolve()
    if not (src / "sglang_omni/models/higgs_tts").is_dir():
        raise SystemExit(f"SGLang-Omni Higgs source not found: {src}")

    sys.path.insert(0, str(src))

    print_kv("sglang_omni_src", str(src))
    print_kv("sglang_omni_commit", git_short_commit(src))

    direct_ok = True
    for name in DIRECT_MODULES:
        status, reason = module_status(name)
        direct_ok = direct_ok and status == "ok"
        print_kv(f"module.{name}", status)
        if reason:
            print_kv(f"module.{name}.reason", normalize_reason(reason))

    full_status, full_reason = module_status(FULL_MODEL_MODULE)
    print_kv(f"module.{FULL_MODEL_MODULE}", full_status)
    if full_reason:
        print_kv(f"module.{FULL_MODEL_MODULE}.reason", normalize_reason(full_reason))

    print_kv("direct_higgs_imports", "ok" if direct_ok else "fail")
    if full_status == "ok":
        print_kv("full_higgs_model_import", "ok")
    else:
        print_kv("full_higgs_model_import", normalize_reason(full_reason))

    if args.require_direct and not direct_ok:
        raise SystemExit(1)
    if args.require_full_model and full_status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
