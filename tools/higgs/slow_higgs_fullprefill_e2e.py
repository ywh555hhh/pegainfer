#!/usr/bin/env python3
"""Slow Higgs Audio full-prefill bring-up decode to a wav.

This is a bring-up tool, not a production runtime. It intentionally stays out
of PegaInfer runtime crates: each audio-codebook step rebuilds the full
sequence as HF Qwen3 ``inputs_embeds`` and runs a full prefill. That is slow,
but it exercises the complete Higgs path from text prompt -> audio code rows ->
codec wav without requiring SGLang-Omni at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "accuracy"))

from dump_higgs_one_step_golden import (  # noqa: E402
    DEFAULT_REVISION,
    HiggsTokenizerAdapter,
    load_backbone,
    load_modality_head_weight,
    load_tokenizer,
)
from higgs_audio_codec import HiggsAudioCodec  # noqa: E402

BOC_ID = 1024
EOC_ID = 1025
CODEC_AUDIO_VOCAB_SIZE = 1024
GREEDY_TEMP_THRESHOLD = 1e-5


@dataclass
class DelayState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None

    def step(self, logits_nv: torch.Tensor, temperature: float) -> torch.Tensor:
        if logits_nv.ndim != 2 or logits_nv.shape[0] != self.num_codebooks:
            raise ValueError(
                f"logits shape {tuple(logits_nv.shape)} does not match "
                f"num_codebooks={self.num_codebooks}"
            )
        if self.generation_done:
            return torch.full(
                (self.num_codebooks,), -1, dtype=torch.long, device=logits_nv.device
            )

        if temperature <= GREEDY_TEMP_THRESHOLD:
            codes = logits_nv.argmax(dim=-1).to(torch.long)
        else:
            probs = (logits_nv / temperature).softmax(dim=-1)
            codes = probs.multinomial(num_samples=1).squeeze(-1).to(torch.long)

        if self.delay_count < self.num_codebooks:
            next_cb = self.delay_count + 1
            if next_cb < self.num_codebooks:
                codes[next_cb:] = BOC_ID
            self.delay_count += 1
        elif self.eoc_countdown is not None:
            self.eoc_countdown -= 1
            if self.eoc_countdown <= 0:
                self.generation_done = True
        elif int(codes[0].item()) == EOC_ID:
            if self.num_codebooks <= 2:
                self.generation_done = True
            else:
                self.eoc_countdown = self.num_codebooks - 2

        if not self.generation_done:
            self.last_codes = codes.clone()
        return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="Hello from PegaInfer.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-wav", required=True, type=Path)
    parser.add_argument("--delayed-codes-out", type=Path)
    parser.add_argument("--codec-input-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
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


def build_inputs_embeds(
    *,
    prompt_ids: list[int],
    delayed_rows: list[torch.Tensor],
    backbone,
    modality_weight: torch.Tensor,
    device: str,
) -> torch.Tensor:
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    text_embeds = backbone.embed_tokens(prompt)
    if not delayed_rows:
        return text_embeds.unsqueeze(0)
    audio_embeds = [
        fused_codebook_embedding(row.to(device=device, dtype=torch.long), modality_weight)
        for row in delayed_rows
    ]
    return torch.cat([text_embeds, torch.stack(audio_embeds, dim=0)], dim=0).unsqueeze(0)


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


def write_json(path: Path | None, value: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_wav(path: Path, samples: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = samples.detach().cpu().float().flatten().tolist()
    pcm = bytearray()
    peak = 0.0
    for value in values:
        if not math.isfinite(value):
            value = 0.0
        value = max(-1.0, min(1.0, float(value)))
        peak = max(peak, abs(value))
        pcm.extend(int(round(value * 32767.0)).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(bytes(pcm))
    return {
        "sample_rate": sample_rate,
        "samples": len(values),
        "duration_sec": len(values) / sample_rate if sample_rate else 0.0,
        "peak_abs": peak,
    }


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    cfg = load_config(args.snapshot_dir)
    text_cfg = dict(cfg["text_config"])
    audio_cfg = dict(cfg["audio_encoder_config"])
    num_codebooks = int(audio_cfg["num_codebooks"])
    vocab_size = int(audio_cfg["vocab_size"])

    adapter = load_tokenizer(args.snapshot_dir, HiggsTokenizerAdapter)
    prompt_ids = adapter.build_prompt(args.prompt)

    model_file = args.snapshot_dir / "model.safetensors"
    backbone = load_backbone(model_file, text_cfg, args.device)
    modality_weight = load_modality_head_weight(model_file, args.device)

    state = DelayState(num_codebooks=num_codebooks)
    delayed_rows_t: list[torch.Tensor] = []
    delayed_rows_json: list[list[int]] = []

    for step_idx in range(args.max_steps):
        inputs_embeds = build_inputs_embeds(
            prompt_ids=prompt_ids,
            delayed_rows=delayed_rows_t,
            backbone=backbone,
            modality_weight=modality_weight,
            device=args.device,
        )
        attention_mask = torch.ones(
            (1, inputs_embeds.shape[1]), dtype=torch.long, device=args.device
        )
        with torch.inference_mode():
            out = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state[:, -1, :].contiguous()
            logits = F.linear(last_hidden, modality_weight).reshape(
                1, num_codebooks, vocab_size
            )[0]
            codes = state.step(logits.to(torch.float32), args.temperature)
        delayed_rows_t.append(codes.detach().clone())
        delayed_rows_json.append([int(x) for x in codes.detach().cpu().tolist()])
        print(f"step={step_idx + 1} codes={delayed_rows_json[-1]}")
        if state.generation_done:
            break

    raw_rows = reverse_delay_pattern(delayed_rows_json, num_codebooks)
    codec_rows = sanitize_codec_rows(raw_rows)
    if not codec_rows:
        raise RuntimeError(
            f"not enough delayed rows for codec input: rows={len(delayed_rows_json)} "
            f"num_codebooks={num_codebooks}"
        )

    write_json(
        args.delayed_codes_out,
        {
            "schema": "higgs-audio-delayed-codes-v1",
            "prompt": args.prompt,
            "num_codebooks": num_codebooks,
            "codebook_size": vocab_size,
            "delayed_codes": delayed_rows_json,
        },
    )
    write_json(
        args.codec_input_out,
        {
            "schema": "higgs-audio-codec-input-v1",
            "num_codebooks": num_codebooks,
            "codebook_size": vocab_size,
            "codec_audio_vocab_size": CODEC_AUDIO_VOCAB_SIZE,
            "raw_codes": codec_rows,
        },
    )

    codec = HiggsAudioCodec.from_pretrained(
        args.snapshot_dir, device=args.device, dtype=torch.float32
    )
    wav = codec.decode(torch.tensor(codec_rows, dtype=torch.long))
    wav_meta = write_wav(args.out_wav, wav, HiggsAudioCodec.SAMPLE_RATE)

    summary = {
        "status": "ok",
        "mode": "slow-fullprefill-local-higgs-codec",
        "snapshot_dir": str(args.snapshot_dir),
        "sglang_omni_runtime_dependency": "false",
        "prompt": args.prompt,
        "device": args.device,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "delayed_rows": len(delayed_rows_json),
        "codec_frames": len(codec_rows),
        "generation_done": state.generation_done,
        "out_wav": str(args.out_wav),
        **wav_meta,
    }
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
