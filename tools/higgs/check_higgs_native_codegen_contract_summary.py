#!/usr/bin/env python3
"""Validate a Higgs Audio native code-generation contract gate summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KEYS = {
    "status",
    "repo",
    "commit",
    "label",
    "sm",
    "nvcc_jobs",
    "gpu_info",
    "cuda_version",
    "nsys_version",
    "ncu_version",
    "rustc_version",
    "cargo_version",
    "uname",
    "contract_report",
    "lib_test_log",
    "runtime_qwen3",
    "runtime_qwen3_log",
    "native_continuation",
    "native_continuation_log",
    "native_continuation_trace",
    "native_continuation_codec",
    "reference_trace",
    "trace_compare",
    "trace_compare_log",
    "forced_prefix",
    "forced_prefix_log",
    "forced_prefix_trace",
    "forced_prefix_codec",
    "forced_prefix_compare",
    "forced_prefix_compare_log",
    "artifact",
    "artifact_sampled_json",
    "artifact_trace_json",
    "artifact_codec_json",
    "artifact_tool_log",
    "profile",
    "native_loop_cmd",
    "nsys_profile",
    "nsys_report",
    "nsys_stats_csv",
    "ncu_profile",
    "ncu_report",
    "ncu_log",
    "claim_boundary",
}

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
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key")
        if key in values:
            raise ValueError(f"{path}:{line_no}: duplicate key {key!r}")
        values[key] = value
    return values


def require_equal(values: dict[str, str], key: str, expected: str | None) -> None:
    if expected is not None and values.get(key) != expected:
        raise ValueError(f"{key}={values.get(key)!r}, expected {expected!r}")


def require_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required artifact missing or empty: {path}")


def require_contains(path: Path, needle: str) -> None:
    if needle not in path.read_text():
        raise ValueError(f"{path} missing expected text: {needle!r}")


def require_json_schema(path: Path, expected: str) -> dict:
    require_nonempty_file(path)
    with path.open() as f:
        value = json.load(f)
    actual = value.get("schema")
    if actual != expected:
        raise ValueError(f"{path} schema={actual!r}, expected {expected!r}")
    return value


def validate(args: argparse.Namespace) -> dict[str, str]:
    summary = args.summary
    if not summary.is_file() or summary.stat().st_size == 0:
        raise ValueError(f"summary missing or empty: {summary}")

    values = parse_summary(summary)
    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"summary missing required keys: {', '.join(missing)}")

    extra = sorted(values.keys() - REQUIRED_KEYS)
    if extra and not args.allow_extra:
        raise ValueError(f"summary has unknown keys: {', '.join(extra)}")

    require_equal(values, "status", "ok")
    require_equal(values, "label", args.expected_label)
    require_equal(values, "sm", args.expected_sm)
    require_equal(values, "nvcc_jobs", args.expected_nvcc_jobs)
    require_equal(values, "claim_boundary", CLAIM_BOUNDARY)

    if not values["commit"]:
        raise ValueError("commit is empty")

    for key in (
        "gpu_info",
        "cuda_version",
        "nsys_version",
        "ncu_version",
        "rustc_version",
        "cargo_version",
        "uname",
    ):
        if not values[key]:
            raise ValueError(f"{key} is empty")

    if not values["sm"].isdigit():
        raise ValueError(f"sm must be numeric, got {values['sm']!r}")

    if not values["nvcc_jobs"].isdigit():
        raise ValueError(f"nvcc_jobs must be numeric, got {values['nvcc_jobs']!r}")

    if values["runtime_qwen3"] not in {"ok", "skipped"}:
        raise ValueError(
            f"runtime_qwen3 must be 'ok' or 'skipped', got {values['runtime_qwen3']!r}"
        )

    if values["native_continuation"] not in {"ok", "skipped"}:
        raise ValueError(
            "native_continuation must be 'ok' or 'skipped', "
            f"got {values['native_continuation']!r}"
        )

    if values["native_continuation"] == "ok" and values["runtime_qwen3"] != "ok":
        raise ValueError("native_continuation=ok requires runtime_qwen3=ok")

    if values["trace_compare"] not in {"ok", "failed", "skipped"}:
        raise ValueError(
            "trace_compare must be 'ok', 'failed', or 'skipped', "
            f"got {values['trace_compare']!r}"
        )

    if values["trace_compare"] == "ok" and values["native_continuation"] != "ok":
        raise ValueError("trace_compare=ok requires native_continuation=ok")

    if values["reference_trace"] and values["trace_compare"] == "skipped":
        raise ValueError("reference_trace was provided but trace_compare was skipped")

    if values["forced_prefix"] not in {"ok", "failed", "skipped"}:
        raise ValueError(
            "forced_prefix must be 'ok', 'failed', or 'skipped', "
            f"got {values['forced_prefix']!r}"
        )

    if values["forced_prefix"] == "ok" and values["native_continuation"] != "ok":
        raise ValueError("forced_prefix=ok requires native_continuation=ok")

    if values["forced_prefix_compare"] not in {"ok", "failed", "skipped"}:
        raise ValueError(
            "forced_prefix_compare must be 'ok', 'failed', or 'skipped', "
            f"got {values['forced_prefix_compare']!r}"
        )

    if values["forced_prefix_compare"] != "skipped" and values["forced_prefix"] != "ok":
        raise ValueError("forced_prefix_compare requires forced_prefix=ok")

    if values["artifact"] != "ok":
        raise ValueError(f"artifact must be 'ok', got {values['artifact']!r}")

    if values["profile"] not in {"ok", "partial", "failed", "skipped"}:
        raise ValueError(
            "profile must be 'ok', 'partial', 'failed', or 'skipped', "
            f"got {values['profile']!r}"
        )

    if values["profile"] != "skipped" and not values["native_loop_cmd"]:
        raise ValueError("profile evidence requires native_loop_cmd")

    if values["nsys_profile"] not in {"ok", "failed", "skipped"}:
        raise ValueError(
            "nsys_profile must be 'ok', 'failed', or 'skipped', "
            f"got {values['nsys_profile']!r}"
        )

    if values["ncu_profile"] not in {
        "ok",
        "failed",
        "blocked:nvgpuctrperm",
        "skipped",
    }:
        raise ValueError(
            "ncu_profile must be 'ok', 'failed', 'blocked:nvgpuctrperm', "
            f"or 'skipped', got {values['ncu_profile']!r}"
        )

    if args.check_files:
        contract_report = Path(values["contract_report"])
        lib_test_log = Path(values["lib_test_log"])
        require_nonempty_file(contract_report)
        require_nonempty_file(lib_test_log)
        require_contains(contract_report, "input: feedback_embedding")
        require_contains(contract_report, "output: final_normed_hidden")
        require_contains(contract_report, "retained_kv: true")
        require_contains(contract_report, "full_prompt_rebuild: false")
        require_contains(lib_test_log, "test result: ok")

        artifact_tool_log = Path(values["artifact_tool_log"])
        require_nonempty_file(Path(values["artifact_sampled_json"]))
        require_nonempty_file(artifact_tool_log)
        require_contains(artifact_tool_log, "higgs prepare codegen artifacts: ok")
        trace = require_json_schema(
            Path(values["artifact_trace_json"]),
            "higgs-audio-native-codegen-trace-v1",
        )
        codec = require_json_schema(
            Path(values["artifact_codec_json"]),
            "higgs-audio-codec-input-v1",
        )
        if len(trace.get("steps", [])) != 8:
            raise ValueError("artifact trace must contain 8 smoke steps")
        if codec.get("raw_codes") != [[1, 102, 203, 304, 405, 506, 607, 708]]:
            raise ValueError("artifact codec rows do not match expected smoke row")

        if values["runtime_qwen3"] == "ok":
            runtime_qwen3_log = Path(values["runtime_qwen3_log"])
            require_nonempty_file(runtime_qwen3_log)
            require_contains(runtime_qwen3_log, "test result: ok")

        if values["native_continuation"] == "ok":
            native_log = Path(values["native_continuation_log"])
            require_nonempty_file(native_log)
            require_contains(native_log, "higgs native continuation smoke: ok")
            require_contains(native_log, "continuation_steps: 8")
            require_contains(native_log, "trace_steps: 9")
            native_trace = require_json_schema(
                Path(values["native_continuation_trace"]),
                "higgs-audio-native-codegen-trace-v1",
            )
            require_json_schema(
                Path(values["native_continuation_codec"]),
                "higgs-audio-codec-input-v1",
            )
            if len(native_trace.get("steps", [])) != 9:
                raise ValueError(
                    "native continuation trace must contain 9 steps "
                    "(one prompt seed plus eight retained continuations)"
                )

        if values["trace_compare"] in {"ok", "failed"}:
            require_nonempty_file(Path(values["reference_trace"]))
            trace_compare_log = Path(values["trace_compare_log"])
            require_nonempty_file(trace_compare_log)
            if values["trace_compare"] == "ok":
                require_contains(trace_compare_log, "higgs codegen trace comparison: ok")
            else:
                require_contains(trace_compare_log, "higgs codegen trace comparison:")

        if values["forced_prefix"] == "ok":
            forced_prefix_log = Path(values["forced_prefix_log"])
            require_nonempty_file(forced_prefix_log)
            require_contains(forced_prefix_log, "higgs native forced-prefix smoke: ok")
            require_contains(forced_prefix_log, "continuation_steps: 8")
            require_contains(forced_prefix_log, "trace_steps: 9")
            forced_trace = require_json_schema(
                Path(values["forced_prefix_trace"]),
                "higgs-audio-native-codegen-trace-v1",
            )
            require_json_schema(
                Path(values["forced_prefix_codec"]),
                "higgs-audio-codec-input-v1",
            )
            if len(forced_trace.get("steps", [])) != 9:
                raise ValueError("forced-prefix trace must contain 9 steps")

        if values["forced_prefix_compare"] in {"ok", "failed"}:
            forced_prefix_compare_log = Path(values["forced_prefix_compare_log"])
            require_nonempty_file(forced_prefix_compare_log)
            if values["forced_prefix_compare"] == "ok":
                require_contains(
                    forced_prefix_compare_log, "higgs codegen trace comparison: ok"
                )
            else:
                require_contains(forced_prefix_compare_log, "higgs codegen trace comparison:")

        if values["nsys_profile"] == "ok":
            require_nonempty_file(Path(values["nsys_report"]))
            if values["nsys_stats_csv"]:
                require_nonempty_file(Path(values["nsys_stats_csv"]))

        if values["ncu_profile"] == "ok":
            require_nonempty_file(Path(values["ncu_report"]))
        elif values["ncu_profile"] == "blocked:nvgpuctrperm":
            require_nonempty_file(Path(values["ncu_log"]))
            require_contains(Path(values["ncu_log"]), "ERR_NVGPUCTRPERM")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-label")
    parser.add_argument("--expected-sm")
    parser.add_argument("--expected-nvcc-jobs")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()

    values = validate(args)
    print(
        "higgs native codegen contract summary: ok "
        f"label={values['label']} "
        f"runtime_qwen3={values['runtime_qwen3']} "
        f"native_continuation={values['native_continuation']} "
        f"trace_compare={values['trace_compare']} "
        f"forced_prefix={values['forced_prefix']} "
        f"forced_prefix_compare={values['forced_prefix_compare']} "
        f"profile={values['profile']} "
        f"claim_boundary={values['claim_boundary']}"
    )


if __name__ == "__main__":
    main()
