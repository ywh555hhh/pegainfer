#!/usr/bin/env python3
"""Validate a PR-ready Higgs Audio 4090 native-codegen evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TRACE_SCHEMA = "higgs-audio-native-codegen-trace-v1"
CODEC_SCHEMA = "higgs-audio-codec-input-v1"
CLAIM_BOUNDARY = "native_contract_only_no_native_decode"


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected key=value, got {raw_line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"{path}:{line_no}: duplicate key {key!r}")
        values[key] = value
    return values


def require(values: dict[str, str], key: str) -> str:
    if key not in values:
        raise ValueError(f"summary missing required key: {key}")
    value = values[key]
    if value == "":
        raise ValueError(f"summary key is empty: {key}")
    return value


def require_not_unavailable(values: dict[str, str], key: str) -> None:
    value = require(values, key)
    if value == "unavailable":
        raise ValueError(f"{key}=unavailable; this is not 4090-ready evidence")


def require_equal(values: dict[str, str], key: str, expected: str) -> None:
    actual = require(values, key)
    if actual != expected:
        raise ValueError(f"{key}={actual!r}, expected {expected!r}")


def require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required artifact missing or empty: {path}")


def require_contains(path: Path, needle: str) -> None:
    text = path.read_text()
    if needle not in text:
        raise ValueError(f"{path} missing expected text: {needle!r}")


def require_json_schema(path: Path, schema: str) -> dict:
    require_nonempty_file(path)
    with path.open() as f:
        value = json.load(f)
    actual = value.get("schema")
    if actual != schema:
        raise ValueError(f"{path} schema={actual!r}, expected {schema!r}")
    return value


def validate(args: argparse.Namespace) -> dict[str, str]:
    summary = args.summary
    require_nonempty_file(summary)
    values = parse_summary(summary)

    if args.expected_label is not None:
        require_equal(values, "label", args.expected_label)
    if args.expected_sm is not None:
        require_equal(values, "sm", args.expected_sm)
    if args.expected_nvcc_jobs is not None:
        require_equal(values, "nvcc_jobs", args.expected_nvcc_jobs)

    require_equal(values, "status", "ok")
    require_equal(values, "claim_boundary", CLAIM_BOUNDARY)
    require_equal(values, "runtime_qwen3", "ok")
    require_equal(values, "native_continuation", "ok")
    require_equal(values, "trace_compare", "ok")
    require_equal(values, "artifact", "ok")
    require_equal(values, "profile", "ok")

    for key in (
        "gpu_info",
        "cuda_version",
        "nsys_version",
        "ncu_version",
        "rustc_version",
        "cargo_version",
        "uname",
    ):
        require_not_unavailable(values, key)

    contract_report = Path(require(values, "contract_report"))
    lib_test_log = Path(require(values, "lib_test_log"))
    runtime_log = Path(require(values, "runtime_qwen3_log"))
    native_log = Path(require(values, "native_continuation_log"))
    compare_log = Path(require(values, "trace_compare_log"))
    pr_evidence = args.pr_evidence

    for path in (contract_report, lib_test_log, runtime_log, native_log, compare_log):
        require_nonempty_file(path)

    require_contains(contract_report, "input: feedback_embedding")
    require_contains(contract_report, "output: final_normed_hidden")
    require_contains(contract_report, "retained_kv: true")
    require_contains(contract_report, "full_prompt_rebuild: false")
    require_contains(lib_test_log, "test result: ok")
    require_contains(runtime_log, "test result: ok")
    require_contains(native_log, "higgs native continuation smoke: ok")
    require_contains(native_log, "continuation_steps: 8")
    require_contains(native_log, "trace_steps: 9")
    require_contains(compare_log, "higgs codegen trace comparison: ok")

    native_trace = require_json_schema(
        Path(require(values, "native_continuation_trace")), TRACE_SCHEMA
    )
    require_json_schema(Path(require(values, "native_continuation_codec")), CODEC_SCHEMA)
    require_json_schema(Path(require(values, "reference_trace")), TRACE_SCHEMA)
    require_nonempty_file(Path(require(values, "nsys_report")))
    require_nonempty_file(Path(require(values, "ncu_report")))

    if len(native_trace.get("steps", [])) != 9:
        raise ValueError("native continuation trace must contain 9 steps")

    require_nonempty_file(pr_evidence)
    require_contains(pr_evidence, "Higgs-Audio native code-generation evidence")
    require_contains(
        pr_evidence,
        "Supported claim: Native incremental audio-code generation diagnostic path with correctness and profiling evidence.",
    )
    require_contains(pr_evidence, "runtime-qwen3 retained bridge tests | passed")
    require_contains(pr_evidence, "Native retained continuation smoke | passed")
    require_contains(pr_evidence, "Official/HF trace comparison | passed")
    require_contains(pr_evidence, "nsys/ncu profiling | passed")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--pr-evidence", required=True, type=Path)
    parser.add_argument("--expected-label")
    parser.add_argument("--expected-sm")
    parser.add_argument("--expected-nvcc-jobs")
    args = parser.parse_args()

    try:
        values = validate(args)
    except ValueError as exc:
        print(f"higgs 4090 evidence bundle: fail: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(
        "higgs 4090 evidence bundle: ok "
        f"label={values['label']} "
        f"sm={values['sm']} "
        f"runtime_qwen3={values['runtime_qwen3']} "
        f"native_continuation={values['native_continuation']} "
        f"trace_compare={values['trace_compare']} "
        f"profile={values['profile']}"
    )


if __name__ == "__main__":
    main()
