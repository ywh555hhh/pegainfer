#!/usr/bin/env python3
"""Decode Higgs Audio codec-input rows into a wav via the local Higgs codec."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path


def load_raw_codes(path: Path) -> list[list[int]]:
    value = json.loads(path.read_text())
    rows = value["raw_codes"] if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("codec input must contain non-empty raw_codes")
    width = len(rows[0])
    if width <= 0:
        raise ValueError("codec input rows must be non-empty")
    for row_idx, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"raw_codes[{row_idx}] width mismatch")
        for col_idx, code in enumerate(row):
            if not isinstance(code, int) or code < 0:
                raise ValueError(
                    f"raw_codes[{row_idx}][{col_idx}] must be a non-negative integer"
                )
    return rows


def write_wav(path: Path, samples, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = samples.detach().cpu().float().flatten().tolist()
    pcm = bytearray()
    for value in values:
        if not math.isfinite(value):
            value = 0.0
        value = max(-1.0, min(1.0, float(value)))
        pcm.extend(int(round(value * 32767.0)).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(bytes(pcm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--codec-input-json", required=True, type=Path)
    parser.add_argument("--out-wav", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    from higgs_audio_codec import HiggsAudioCodec

    rows = load_raw_codes(args.codec_input_json)
    codes = torch.tensor(rows, dtype=torch.long)
    codec = HiggsAudioCodec.from_pretrained(
        args.model_dir, device=args.device, dtype=torch.float32
    )
    waveform = codec.decode(codes)
    write_wav(args.out_wav, waveform, HiggsAudioCodec.SAMPLE_RATE)

    print("higgs codec sidecar: ok")
    print(f"  codec_frames: {len(rows)}")
    print(f"  samples: {int(waveform.numel())}")
    print(f"  sample_rate: {HiggsAudioCodec.SAMPLE_RATE}")
    print(f"  out_wav: {args.out_wav}")


if __name__ == "__main__":
    main()
