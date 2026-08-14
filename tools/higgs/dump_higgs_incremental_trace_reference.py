#!/usr/bin/env python3
"""Dump an official/HF-style Higgs Audio incremental code-generation trace.

This is a reference/golden-generation tool only. It uses Transformers Qwen3
with ``use_cache=True`` / ``past_key_values`` for the text-body continuation
and feeds each audio step as the fused codebook embedding. It must not be
called from PegaInfer runtime code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "accuracy"))

TRACE_SCHEMA = "higgs-audio-native-codegen-trace-v1"
CODEC_INPUT_SCHEMA = "higgs-audio-codec-input-v1"
DEFAULT_MODEL_ID = "bosonai/higgs-tts-3-4b"
DEFAULT_REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"
DEFAULT_PROMPT = "Hello from PegaInfer."
BOC_ID = 1024
EOC_ID = 1025
CODEC_AUDIO_VOCAB_SIZE = 1024
TOP_K = 64

torch: Any = None
F: Any = None


class DelayState:
    def __init__(self, num_codebooks: int) -> None:
        self.num_codebooks = num_codebooks
        self.delay_count = 0
        self.eoc_countdown: int | None = None
        self.generation_done = False

    def step(self, logits_nv: torch.Tensor) -> tuple[list[int], list[int] | None, bool]:
        if logits_nv.ndim != 2 or logits_nv.shape[0] != self.num_codebooks:
            raise ValueError(
                f"logits shape {tuple(logits_nv.shape)} does not match "
                f"num_codebooks={self.num_codebooks}"
            )
        if self.generation_done:
            return [-1] * self.num_codebooks, None, True

        sampled = logits_nv.argmax(dim=-1).to(torch.long)
        delayed = sampled.clone()
        if self.delay_count < self.num_codebooks:
            next_cb = self.delay_count + 1
            if next_cb < self.num_codebooks:
                delayed[next_cb:] = BOC_ID
            self.delay_count += 1
        elif self.eoc_countdown is not None:
            self.eoc_countdown -= 1
            if self.eoc_countdown <= 0:
                self.generation_done = True
        elif int(delayed[0].item()) == EOC_ID:
            if self.num_codebooks <= 2:
                self.generation_done = True
            else:
                self.eoc_countdown = self.num_codebooks - 2

        if self.generation_done:
            return (
                [int(x) for x in sampled.detach().cpu().tolist()],
                None,
                True,
            )
        return (
            [int(x) for x in sampled.detach().cpu().tolist()],
            [int(x) for x in delayed.detach().cpu().tolist()],
            False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--codec-input-out", type=Path)
    parser.add_argument(
        "--sglang-omni-src",
        default="",
        help="Optional SGLang-Omni source tree for tokenizer/head semantics.",
    )
    parser.add_argument(
        "--require-sglang-omni-source",
        action="store_true",
        help="Fail if --sglang-omni-src cannot provide Higgs reference modules.",
    )
    return parser.parse_args()


def load_config(snapshot_dir: Path) -> dict[str, Any]:
    with (snapshot_dir / "config.json").open() as f:
        return json.load(f)


def fused_codebook_embedding(codes_n: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    num_codebooks = int(codes_n.shape[0])
    vocab_size = int(weight.shape[0] // num_codebooks)
    offsets = torch.arange(num_codebooks, device=codes_n.device, dtype=codes_n.dtype)
    fused_ids = codes_n + offsets * vocab_size
    return F.embedding(fused_ids, weight).sum(dim=0)


def reverse_delay_pattern(rows: list[list[int]], num_codebooks: int) -> list[list[int]]:
    frame_count = len(rows) - num_codebooks + 1
    if frame_count <= 0:
        return []
    out: list[list[int]] = []
    for frame_idx in range(frame_count):
        out.append([int(rows[frame_idx + cb][cb]) for cb in range(num_codebooks)])
    return out


def sanitize_codec_rows(rows: list[list[int]]) -> list[list[int]]:
    return [
        [0 if int(code) >= CODEC_AUDIO_VOCAB_SIZE else int(code) for code in row]
        for row in rows
    ]


def logits_summary(logits_nv: torch.Tensor) -> dict[str, Any]:
    logits = logits_nv.detach().to(torch.float32)
    argmax = logits.argmax(dim=-1).to(torch.int64)
    top_ids = torch.topk(torch.log_softmax(logits, dim=-1), k=TOP_K, dim=-1).indices
    l2_norm = torch.linalg.vector_norm(logits.reshape(-1)).item()
    return {
        "argmax": [int(x) for x in argmax.cpu().tolist()],
        "top_ids": [[int(x) for x in row] for row in top_ids.cpu().tolist()],
        "logits": [[float(x) for x in row] for row in logits.cpu().tolist()],
        "top64_min_overlap_with_previous": None,
        "logits_l2_norm": float(l2_norm),
    }


def write_json(path: Path | None, value: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    global F, torch
    import torch as torch_module
    import torch.nn.functional as functional_module
    from dump_higgs_one_step_golden import (  # noqa: E402
        HiggsTokenizerAdapter,
        load_backbone,
        load_modality_head_weight,
        load_sglang_omni_reference,
        load_tokenizer,
    )

    torch = torch_module
    F = functional_module

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    cfg = load_config(args.snapshot_dir)
    text_cfg = dict(cfg["text_config"])
    audio_cfg = dict(cfg["audio_encoder_config"])
    num_codebooks = int(audio_cfg["num_codebooks"])

    reference = load_sglang_omni_reference(
        args.sglang_omni_src,
        required=args.require_sglang_omni_source,
    )
    adapter_cls = reference.get("tokenizer_adapter_cls", HiggsTokenizerAdapter)
    adapter = load_tokenizer(args.snapshot_dir, adapter_cls)
    prompt_ids = adapter.build_prompt(args.prompt)

    model_file = args.snapshot_dir / "model.safetensors"
    backbone = load_backbone(model_file, text_cfg, args.device)
    modality_weight = load_modality_head_weight(model_file, args.device)

    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    attention_mask = torch.ones_like(prompt, dtype=torch.long, device=args.device)
    delayed_rows: list[list[int]] = []
    trace_steps: list[dict[str, Any]] = []
    delay_state = DelayState(num_codebooks)

    with torch.inference_mode():
        prefill = backbone(
            input_ids=prompt,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prefill.past_key_values
        last_hidden = prefill.last_hidden_state[:, -1, :].contiguous()
        logits = F.linear(last_hidden, modality_weight).reshape(
            1, num_codebooks, int(audio_cfg["vocab_size"])
        )[0]
        sampled, delayed, done = delay_state.step(logits)
        if delayed is not None:
            delayed_rows.append(delayed)
        trace_steps.append(
            {
                "step": 0,
                "sampled_codes": sampled,
                "delayed_codes": delayed,
                "raw_codes": sanitize_codec_rows(
                    reverse_delay_pattern(delayed_rows, num_codebooks)
                ),
                "generation_done": done,
                "logits": logits_summary(logits),
            }
        )

        for step_idx in range(1, args.steps + 1):
            if delay_state.generation_done or not delayed_rows:
                break
            feedback = fused_codebook_embedding(
                torch.tensor(delayed_rows[-1], dtype=torch.long, device=args.device),
                modality_weight,
            ).reshape(1, 1, -1)
            attention_mask = torch.ones(
                (1, len(prompt_ids) + step_idx), dtype=torch.long, device=args.device
            )
            out = backbone(
                inputs_embeds=feedback,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            last_hidden = out.last_hidden_state[:, -1, :].contiguous()
            logits = F.linear(last_hidden, modality_weight).reshape(
                1, num_codebooks, int(audio_cfg["vocab_size"])
            )[0]
            sampled, delayed, done = delay_state.step(logits)
            if delayed is not None:
                delayed_rows.append(delayed)
            trace_steps.append(
                {
                    "step": step_idx,
                    "sampled_codes": sampled,
                    "delayed_codes": delayed,
                    "raw_codes": sanitize_codec_rows(
                        reverse_delay_pattern(delayed_rows, num_codebooks)
                    ),
                    "generation_done": done,
                    "logits": logits_summary(logits),
                }
            )

    raw_rows = sanitize_codec_rows(reverse_delay_pattern(delayed_rows, num_codebooks))
    trace = {
        "schema": TRACE_SCHEMA,
        "prompt_tokens": len(prompt_ids),
        "reference": {
            "kind": "hf_transformers_incremental_past_key_values",
            "model_id": args.model_id,
            "revision": args.revision,
            "snapshot_dir": str(args.snapshot_dir),
            "prompt": args.prompt,
            "device": args.device,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "sglang_omni_source_dir": reference.get("source_dir", ""),
            "sglang_omni_source_commit": reference.get("source_commit", ""),
        },
        "steps": trace_steps,
    }
    write_json(args.out, trace)
    write_json(
        args.codec_input_out,
        {
            "schema": CODEC_INPUT_SCHEMA,
            "num_codebooks": num_codebooks,
            "codebook_size": int(audio_cfg["vocab_size"]),
            "codec_audio_vocab_size": CODEC_AUDIO_VOCAB_SIZE,
            "raw_codes": raw_rows,
        },
    )
    print("higgs incremental trace reference: ok")
    print(f"  prompt_tokens: {len(prompt_ids)}")
    print(f"  trace_steps: {len(trace_steps)}")
    print(f"  raw_codec_rows: {len(raw_rows)}")
    print(f"  out: {args.out}")
    if args.codec_input_out:
        print(f"  codec_input_out: {args.codec_input_out}")


if __name__ == "__main__":
    main()
