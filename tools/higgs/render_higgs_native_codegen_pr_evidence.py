#!/usr/bin/env python3
"""Render PR-ready Higgs Audio native codegen evidence from a gate summary."""

from __future__ import annotations

import argparse
from pathlib import Path


CLAIM_BOUNDARY = "native_contract_only_no_native_decode"
NON_CLAIMS = (
    "native wav E2E",
    "native codec/vocoder",
    "production serving",
    "strict trace parity without the 4090 trace-comparison table",
)


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
    value = values.get(key)
    if value is None:
        raise ValueError(f"summary missing required key: {key}")
    return value


def status_label(status: str) -> str:
    if status == "ok":
        return "passed"
    if status == "skipped":
        return "not run"
    return status or "unknown"


def artifact(value: str) -> str:
    return value if value else "n/a"


def supported_claim(values: dict[str, str]) -> tuple[str, list[str]]:
    runtime_ok = require(values, "runtime_qwen3") == "ok"
    native_ok = require(values, "native_continuation") == "ok"
    trace_ok = require(values, "trace_compare") == "ok"
    profile_ok = require(values, "profile") == "ok"
    nsys_ok = values.get("nsys_profile") == "ok"
    forced_prefix_ok = values.get("forced_prefix") == "ok"

    if trace_ok and profile_ok:
        return (
            "Native incremental audio-code generation diagnostic path with correctness and profiling evidence.",
            [
                "Runtime-Qwen3 retained bridge passed.",
                "Native retained continuation smoke emitted the expected trace and codec artifacts.",
                "Official/HF incremental reference trace comparison passed.",
                "4090 `nsys` and `ncu` profiling artifacts were captured.",
            ],
        )
    if trace_ok:
        return (
            "Native incremental audio-code generation diagnostic path with measured semantic trace evidence, but profiling evidence is still missing.",
            [
                "Runtime-Qwen3 retained bridge passed.",
                "Native retained continuation smoke emitted the expected trace and codec artifacts.",
                "Official/HF incremental reference trace comparison passed.",
                "4090 `nsys`/`ncu` profiling evidence has not been captured.",
            ],
        )
    if runtime_ok and native_ok and forced_prefix_ok and nsys_ok:
        return (
            "Native retained-continuation execution with forced common-prefix diagnostic evidence and 4090 nsys profiling; strict semantic parity and ncu profiling are still unresolved.",
            [
                "Runtime-Qwen3 retained bridge passed.",
                "Native retained continuation smoke emitted a retained-KV trace.",
                "Forced common-prefix replay ran the native retained-KV/audio-head path while consuming HF sampled rows.",
                "4090 `nsys` profiling artifacts were captured.",
                "Official/HF free-running trace comparison has not passed.",
                "Do not claim strict semantic parity or full profiler coverage yet.",
            ],
        )
    if runtime_ok and native_ok and forced_prefix_ok:
        return (
            "Native retained-continuation execution with forced common-prefix diagnostic evidence; strict semantic parity and profiling are still unresolved.",
            [
                "Runtime-Qwen3 retained bridge passed.",
                "Native retained continuation smoke emitted a retained-KV trace.",
                "Forced common-prefix replay ran the native retained-KV/audio-head path while consuming HF sampled rows.",
                "Official/HF free-running trace comparison has not passed.",
                "4090 `nsys`/`ncu` profiling evidence has not been captured in this summary.",
            ],
        )
    if runtime_ok and native_ok and nsys_ok:
        return (
            "Native retained-continuation execution with 4090 nsys profiling; strict semantic parity and ncu profiling are still unresolved.",
            [
                "Runtime-Qwen3 retained bridge passed.",
                "Native retained continuation smoke emitted a retained-KV trace.",
                "4090 `nsys` profiling artifacts were captured.",
                "Official/HF incremental reference trace comparison has not passed.",
                "Do not claim strict semantic parity or full profiler coverage yet.",
            ],
        )
    if runtime_ok and native_ok:
        return (
            "Native retained-continuation execution only; semantic parity and profiling are unproven.",
            [
                "Runtime-Qwen3 retained bridge passed.",
                "Native retained continuation smoke emitted a retained-KV trace.",
                "Official/HF incremental reference trace comparison has not passed.",
                "Do not claim semantic parity from sampled rows alone.",
            ],
        )
    return (
        "Bring-up scaffolding only; no native retained decode claim.",
        [
            "Local contract and artifact-schema checks may pass without GPU execution.",
            "Runtime-Qwen3 retained bridge has not passed on the target Linux/4090 host.",
            "Native continuation, reference trace comparison, and profiling are not established.",
        ],
    )


def render(values: dict[str, str]) -> str:
    claim_boundary = require(values, "claim_boundary")
    if claim_boundary != CLAIM_BOUNDARY:
        raise ValueError(
            f"unexpected claim boundary {claim_boundary!r}; expected {CLAIM_BOUNDARY!r}"
        )
    claim, claim_reasons = supported_claim(values)

    lines: list[str] = []
    lines.append("## Higgs-Audio native code-generation evidence")
    lines.append("")
    lines.append(
        "Scope: feature-gated Higgs-Audio bring-up for native incremental audio-code generation diagnostics."
    )
    lines.append(
        f"Claim boundary: `{claim_boundary}`. This does not claim {', '.join(NON_CLAIMS)}."
    )
    lines.append(f"Supported claim: {claim}")
    lines.append("")
    lines.append("### Claim decision")
    lines.append("")
    for reason in claim_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("### Design boundary")
    lines.append("")
    lines.append(
        "- Higgs-Audio owns the multimodal continuation semantics: feedback embedding, audio head, sampling, delay pattern, raw codec rows, and trace rows."
    )
    lines.append(
        "- The Qwen3 touch is intentionally narrow: an embedding-fed retained-KV diagnostic step that returns final normed hidden before the text `lm_head`."
    )
    lines.append(
        "- The current slice does not require shared `pegainfer-core` or shared `pegainfer-kernels` semantic changes; Python remains reference/golden/profiling tooling only."
    )
    lines.append(
        "- Copying the Qwen3 decode body into Higgs would keep the diff model-local, but would duplicate KV, numeric policy, CUDA graph, and future Qwen3 decode fixes."
    )
    lines.append("")
    lines.append("### Environment")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key in (
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
    ):
        lines.append(f"| `{key}` | `{require(values, key)}` |")
    lines.append("")
    lines.append("### Gates")
    lines.append("")
    lines.append("| Gate | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Continuation contract | passed | `{artifact(require(values, 'contract_report'))}` |"
    )
    lines.append(f"| Higgs lib tests | passed | `{artifact(require(values, 'lib_test_log'))}` |")
    lines.append(
        f"| runtime-qwen3 retained bridge tests | {status_label(require(values, 'runtime_qwen3'))} | `{artifact(require(values, 'runtime_qwen3_log'))}` |"
    )
    lines.append(
        f"| Native retained continuation smoke | {status_label(require(values, 'native_continuation'))} | `{artifact(require(values, 'native_continuation_log'))}` |"
    )
    lines.append(
        f"| Official/HF trace comparison | {status_label(require(values, 'trace_compare'))} | `{artifact(require(values, 'trace_compare_log'))}` |"
    )
    if "forced_prefix" in values:
        lines.append(
            f"| Forced common-prefix smoke | {status_label(values.get('forced_prefix', 'skipped'))} | `{artifact(values.get('forced_prefix_log', ''))}` |"
        )
        lines.append(
            f"| Forced common-prefix comparison | {status_label(values.get('forced_prefix_compare', 'skipped'))} | `{artifact(values.get('forced_prefix_compare_log', ''))}` |"
        )
    lines.append(
        f"| nsys/ncu profiling | {status_label(require(values, 'profile'))} | `nsys={artifact(require(values, 'nsys_report'))}` / `ncu={artifact(require(values, 'ncu_report'))}` |"
    )
    if "nsys_profile" in values or "ncu_profile" in values:
        lines.append(
            f"| nsys profile | {status_label(values.get('nsys_profile', 'skipped'))} | `{artifact(values.get('nsys_report', ''))}` |"
        )
        lines.append(
            f"| nsys stats | {status_label('ok' if values.get('nsys_profile') == 'ok' and values.get('nsys_stats_csv') else 'skipped')} | `{artifact(values.get('nsys_stats_csv', ''))}` |"
        )
        lines.append(
            f"| ncu profile | {status_label(values.get('ncu_profile', 'skipped'))} | `{artifact(values.get('ncu_log') or values.get('ncu_report', ''))}` |"
        )
    lines.append("")
    lines.append("### Artifacts")
    lines.append("")
    lines.append("| Artifact | Path |")
    lines.append("| --- | --- |")
    for key in (
        "artifact_trace_json",
        "artifact_codec_json",
        "native_continuation_trace",
        "native_continuation_codec",
        "forced_prefix_trace",
        "forced_prefix_codec",
        "reference_trace",
    ):
        if key in values:
            lines.append(f"| `{key}` | `{artifact(require(values, key))}` |")
    lines.append("")
    lines.append("### Remaining limitations")
    lines.append("")
    limitations = []
    if require(values, "native_continuation") != "ok":
        limitations.append(
            "Native retained continuation smoke has not passed on the target Linux/4090 host."
        )
    if require(values, "trace_compare") != "ok":
        limitations.append(
            "Official/HF incremental reference trace comparison has not passed; sampled-row-only traces are not semantic parity evidence."
        )
    if values.get("forced_prefix") == "ok" and values.get("forced_prefix_compare") != "ok":
        limitations.append(
            "Forced common-prefix replay improves diagnosis but still does not satisfy strict trace parity."
        )
    if values.get("nsys_profile") == "ok" and values.get("ncu_profile") != "ok":
        limitations.append(
            "4090 nsys profiling was captured, but ncu profiling did not complete."
        )
    elif require(values, "profile") != "ok":
        limitations.append("4090 nsys/ncu profiling evidence has not been fully captured.")
    if not limitations:
        limitations.append(
            "This still remains below native wav E2E because codec/vocoder and production serving are out of scope."
        )
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    text = render(parse_summary(args.summary))
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
