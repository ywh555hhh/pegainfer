#!/usr/bin/env python3
"""Request a Higgs Audio /v1/audio/speech wav and emit validation metadata."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def optional_json_file(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text())


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "voice": args.voice,
        "input": args.input,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.seed is not None:
        payload["seed"] = args.seed
    references = optional_json_file(args.references_json)
    if references is not None:
        payload["references"] = references
    return payload


def request_wav(api_base: str, payload: dict[str, Any], timeout: float) -> tuple[bytes, str]:
    endpoint = api_base.rstrip("/") + "/v1/audio/speech"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to connect to {endpoint}: {exc}") from exc


def inspect_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    if sample_width != 2:
        peak = None
    else:
        peak_i16 = 0
        for offset in range(0, len(raw), 2):
            sample = int.from_bytes(raw[offset : offset + 2], "little", signed=True)
            peak_i16 = max(peak_i16, abs(sample))
        peak = peak_i16 / 32767.0
    duration = frames / sample_rate if sample_rate else math.nan
    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_sec": duration,
        "peak_abs": peak,
    }


def require_valid_audio(metadata: dict[str, Any], min_duration_sec: float) -> None:
    if metadata["channels"] != 1:
        raise RuntimeError(f"expected mono wav, got channels={metadata['channels']}")
    if metadata["sample_width"] != 2:
        raise RuntimeError(
            f"expected 16-bit wav, got sample_width={metadata['sample_width']}"
        )
    if metadata["sample_rate"] != 24_000:
        raise RuntimeError(
            f"expected Higgs 24 kHz wav, got sample_rate={metadata['sample_rate']}"
        )
    if metadata["duration_sec"] < min_duration_sec:
        raise RuntimeError(
            f"wav too short: duration_sec={metadata['duration_sec']:.3f}, "
            f"min_duration_sec={min_duration_sec:.3f}"
        )
    peak = metadata["peak_abs"]
    if peak is None or peak <= 0.0:
        raise RuntimeError("wav appears silent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--model", default="bosonai/higgs-audio-v3-tts-4b")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-wav", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--payload-out", type=Path)
    parser.add_argument("--references-json", type=Path)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=positive_int)
    parser.add_argument("--max-new-tokens", type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--min-duration-sec", type=float, default=0.2)
    args = parser.parse_args()

    payload = build_payload(args)
    audio, content_type = request_wav(args.api_base, payload, args.timeout_sec)
    if not audio:
        raise RuntimeError("empty audio response")

    args.out_wav.parent.mkdir(parents=True, exist_ok=True)
    args.out_wav.write_bytes(audio)
    if args.payload_out is not None:
        args.payload_out.parent.mkdir(parents=True, exist_ok=True)
        args.payload_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    metadata = inspect_wav(args.out_wav)
    require_valid_audio(metadata, args.min_duration_sec)

    lines = [
        "status=ok",
        f"api_base={args.api_base}",
        f"model={args.model}",
        f"voice={args.voice}",
        f"out_wav={args.out_wav}",
        f"content_type={content_type}",
        f"channels={metadata['channels']}",
        f"sample_width={metadata['sample_width']}",
        f"sample_rate={metadata['sample_rate']}",
        f"frames={metadata['frames']}",
        f"duration_sec={metadata['duration_sec']:.6f}",
        f"peak_abs={metadata['peak_abs']:.6f}",
    ]
    text = "\n".join(lines) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
