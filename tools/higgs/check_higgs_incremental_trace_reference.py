#!/usr/bin/env python3
"""Validate a Higgs Audio official/HF incremental reference trace JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "higgs-audio-native-codegen-trace-v1"
REFERENCE_KIND = "hf_transformers_incremental_past_key_values"
NUM_CODEBOOKS = 8
TOP_K = 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_int(value: Any, field: str) -> int:
    require(isinstance(value, int), f"{field} must be int")
    return value


def require_list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be list")
    return value


def require_code_row(value: Any, field: str) -> list[int]:
    row = require_list(value, field)
    require(len(row) == NUM_CODEBOOKS, f"{field} must have {NUM_CODEBOOKS} codebooks")
    for idx, item in enumerate(row):
        require(isinstance(item, int), f"{field}[{idx}] must be int")
    return row


def require_optional_code_row(value: Any, field: str) -> list[int] | None:
    if value is None:
        return None
    return require_code_row(value, field)


def validate_logits(value: Any, step_idx: int) -> None:
    require(isinstance(value, dict), f"steps[{step_idx}].logits must be object")
    argmax = require_code_row(value.get("argmax"), f"steps[{step_idx}].logits.argmax")
    top_ids = require_list(value.get("top_ids"), f"steps[{step_idx}].logits.top_ids")
    logits = require_list(value.get("logits"), f"steps[{step_idx}].logits.logits")
    require(len(top_ids) == NUM_CODEBOOKS, f"steps[{step_idx}].logits.top_ids must have {NUM_CODEBOOKS} rows")
    require(len(logits) == NUM_CODEBOOKS, f"steps[{step_idx}].logits.logits must have {NUM_CODEBOOKS} rows")
    for codebook, row in enumerate(top_ids):
        row = require_list(row, f"steps[{step_idx}].logits.top_ids[{codebook}]")
        require(len(row) >= TOP_K, f"steps[{step_idx}].logits.top_ids[{codebook}] must include top-{TOP_K}")
        for item_idx, item in enumerate(row[:TOP_K]):
            require(
                isinstance(item, int),
                f"steps[{step_idx}].logits.top_ids[{codebook}][{item_idx}] must be int",
            )
    for codebook, row in enumerate(logits):
        row = require_list(row, f"steps[{step_idx}].logits.logits[{codebook}]")
        require(row, f"steps[{step_idx}].logits.logits[{codebook}] must be non-empty")
        for item_idx, item in enumerate(row):
            require(
                isinstance(item, int | float),
                f"steps[{step_idx}].logits.logits[{codebook}][{item_idx}] must be number",
            )
        argmax_id = argmax[codebook]
        require(
            0 <= argmax_id < len(row),
            f"steps[{step_idx}].logits.argmax[{codebook}] out of range for logits row",
        )
    require(
        isinstance(value.get("logits_l2_norm"), int | float),
        f"steps[{step_idx}].logits.logits_l2_norm must be number",
    )


def validate(path: Path, *, expected_steps: int | None) -> dict[str, Any]:
    with path.open() as f:
        value = json.load(f)
    require(value.get("schema") == TRACE_SCHEMA, f"schema must be {TRACE_SCHEMA}")
    require_int(value.get("prompt_tokens"), "prompt_tokens")
    reference = value.get("reference")
    require(isinstance(reference, dict), "reference must be object")
    require(
        reference.get("kind") == REFERENCE_KIND,
        f"reference.kind must be {REFERENCE_KIND}",
    )
    for key in ("model_id", "revision", "snapshot_dir", "prompt", "device", "torch", "transformers"):
        require(isinstance(reference.get(key), str) and reference[key], f"reference.{key} must be non-empty string")

    steps = require_list(value.get("steps"), "steps")
    require(steps, "steps must be non-empty")
    if expected_steps is not None:
        require(
            len(steps) == expected_steps,
            f"steps length {len(steps)} does not match expected {expected_steps}",
        )
    for idx, step in enumerate(steps):
        require(isinstance(step, dict), f"steps[{idx}] must be object")
        require(require_int(step.get("step"), f"steps[{idx}].step") == idx, f"steps[{idx}].step must equal {idx}")
        require_code_row(step.get("sampled_codes"), f"steps[{idx}].sampled_codes")
        require_optional_code_row(step.get("delayed_codes"), f"steps[{idx}].delayed_codes")
        raw_rows = require_list(step.get("raw_codes"), f"steps[{idx}].raw_codes")
        for row_idx, row in enumerate(raw_rows):
            require_code_row(row, f"steps[{idx}].raw_codes[{row_idx}]")
        require(isinstance(step.get("generation_done"), bool), f"steps[{idx}].generation_done must be bool")
        validate_logits(step.get("logits"), idx)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()

    value = validate(args.trace, expected_steps=args.expected_steps)
    print(
        "higgs incremental reference trace: ok "
        f"steps={len(value['steps'])} "
        f"prompt_tokens={value['prompt_tokens']} "
        f"kind={value['reference']['kind']}"
    )


if __name__ == "__main__":
    main()
