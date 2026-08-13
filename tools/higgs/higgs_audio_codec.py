"""Local Higgs Audio V2 codec facade used by PegaInfer Higgs bring-up tools.

This module is a small, Higgs-owned port of the SGLang-Omni
``HiggsAudioCodec`` decode path. It intentionally does not import
``sglang_omni`` at runtime: SGLang-Omni is the reference source for these
semantics, while the PegaInfer Higgs tooling owns this copy.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

_CODEC_IN_TTS_CKPT_PREFIX = "tied.embedding.modality_embeddings.0.model."
_BUNDLED_CODEC_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "higgs_audio_v2_tokenizer.json"
)


def _resolve_ckpt_dir(model_path: str | Path) -> str:
    path = str(model_path)
    if os.path.isdir(path):
        return path
    return snapshot_download(path)


def _load_codec_state_dict(tts_ckpt_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(tts_ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for full_name, shard in weight_map.items():
            if full_name.startswith(_CODEC_IN_TTS_CKPT_PREFIX):
                shards.setdefault(shard, []).append(full_name)
    else:
        shards = {"model.safetensors": []}

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        shard_path = os.path.join(tts_ckpt_dir, shard)
        with safe_open(shard_path, framework="pt") as f:
            keys = names or [
                key for key in f.keys() if key.startswith(_CODEC_IN_TTS_CKPT_PREFIX)
            ]
            for full_name in keys:
                state[full_name[len(_CODEC_IN_TTS_CKPT_PREFIX) :]] = f.get_tensor(
                    full_name
                )
    return state


class HiggsAudioCodec:
    """Frozen decode wrapper around ``HiggsAudioV2TokenizerModel``."""

    SAMPLE_RATE: int = 24_000

    def __init__(
        self, model: HiggsAudioV2TokenizerModel, *, device: torch.device
    ) -> None:
        self.model = model
        self.device = device
        self._decode_single_flight_lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HiggsAudioCodec":
        device = torch.device(device)
        ckpt_dir = _resolve_ckpt_dir(model_path)

        config = HiggsAudioV2TokenizerConfig.from_json_file(
            str(_BUNDLED_CODEC_CONFIG_PATH)
        )
        model = HiggsAudioV2TokenizerModel(config).to(dtype=dtype).eval()

        state = _load_codec_state_dict(ckpt_dir)
        if not state:
            raise FileNotFoundError(
                f"No codec weights found under {_CODEC_IN_TTS_CKPT_PREFIX!r} in "
                f"{ckpt_dir}; this checkpoint does not bundle the audio codec."
            )

        missing, _unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"Codec weight load is too sparse: {len(missing)} missing / "
                f"{len(state)} loaded; bundled codec config may be incompatible "
                "with the installed Transformers version."
            )

        model = model.to(device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, device=device)

    @torch.no_grad()
    def decode(self, codes_tn: torch.Tensor) -> torch.Tensor:
        """Decode ``[T, num_codebooks]`` integer codec codes into mono ``[L]``."""
        if codes_tn.ndim != 2:
            raise ValueError(
                f"codes must be 2-D [T, num_codebooks], got {tuple(codes_tn.shape)}"
            )
        with self._decode_single_flight_lock:
            codes_bnt = codes_tn.transpose(0, 1).unsqueeze(0)
            audio = self.model.decode(
                codes_bnt.to(device=self.device, dtype=torch.long)
            ).audio_values
            return audio.squeeze(0).squeeze(0).cpu()


__all__ = ["HiggsAudioCodec"]
