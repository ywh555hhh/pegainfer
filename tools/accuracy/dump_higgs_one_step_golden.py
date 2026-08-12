#!/usr/bin/env python3
"""Dump a Higgs Audio one-step audio-logits golden.

This generator intentionally uses the SGLang-Omni Higgs prompt/head contract
without importing the full SGLang server stack. The prompt builder mirrors
`sglang_omni.models.higgs_tts.text_tokenizer.HiggsTokenizerAdapter`; the fused
audio head mirrors `HiggsFusedMultiTextHead.generate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

AUDIO_PLACEHOLDER_ID = -100
REQUIRED_SPECIALS = ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>")
DEFAULT_MODEL_ID = "bosonai/higgs-tts-3-4b"
DEFAULT_REVISION = "7556c17e05201fccd9c8cc120bc216dcc7b5d561"
DEFAULT_PROMPTS = ("Hello from PegaInfer.",)


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id = int(vocab["<|tts|>"])
        self.ref_audio_id = int(vocab["<|ref_audio|>"])
        self.text_id = int(vocab["<|text|>"])
        self.audio_id = int(vocab["<|audio|>"])
        self.ref_text_id = vocab.get("<|ref_text|>")

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
    ) -> list[int]:
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(int(self.ref_text_id))
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return [int(x) for x in ids]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def remap_backbone_key(src: str) -> str | None:
    if src == "tied.embedding.text_embedding.weight":
        return "embed_tokens.weight"
    if src.startswith("body.layers."):
        return src.removeprefix("body.")
    if src.startswith("body.norm."):
        return src.removeprefix("body.")
    return None


def load_backbone(model_file: Path, text_cfg: dict[str, Any], device: str) -> Qwen3Model:
    cfg = Qwen3Config(**text_cfg)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3Model(cfg)
    model.to_empty(device=device)
    # to_empty() does not populate non-persistent RoPE buffers created on the
    # meta device, so rebuild rotary_emb on the real device before loading params.
    model.rotary_emb = Qwen3RotaryEmbedding(cfg, device=device)
    model.to(dtype=torch.bfloat16)
    model.eval()
    params = dict(model.named_parameters())
    loaded: set[str] = set()
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        for src in f.keys():
            dst = remap_backbone_key(src)
            if dst is None:
                continue
            if dst not in params:
                raise KeyError(f"remapped key {src} -> {dst}, but Qwen3Model has no such parameter")
            p = params[dst]
            t = f.get_tensor(src)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch {src}->{dst}: ckpt {tuple(t.shape)} vs model {tuple(p.shape)}")
            p.data.copy_(t.to(device=device, dtype=p.dtype, non_blocking=False))
            loaded.add(dst)
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"missing {len(missing)} backbone parameters, first: {missing[:8]}")
    return model


def load_modality_head_weight(model_file: Path, device: str) -> torch.Tensor:
    key = "tied.embedding.modality_embeddings.0.embedding.weight"
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"missing fused modality embedding/head weight {key}")
        weight = f.get_tensor(key)
    if tuple(weight.shape) != (8208, 2560):
        raise ValueError(f"unexpected modality head weight shape {tuple(weight.shape)}")
    return weight.to(device=device, dtype=torch.bfloat16)


def load_tokenizer(snapshot_dir: Path) -> HiggsTokenizerAdapter:
    raw = Tokenizer.from_file(str(snapshot_dir / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    return HiggsTokenizerAdapter(tokenizer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--snapshot-dir", default="")
    ap.add_argument("--out", default="test_data/higgs-one-step-audio-logits.safetensors")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
    else:
        local_dir = f"models/higgs-tts-3-4b-{args.revision}"
        if args.download or not Path(local_dir, "model.safetensors").exists():
            snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        else:
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
                hf_hub_download(args.model_id, name, revision=args.revision)
        snapshot_dir = Path(local_dir)

    config_path = snapshot_dir / "config.json"
    tokenizer_path = snapshot_dir / "tokenizer.json"
    model_file = snapshot_dir / "model.safetensors"
    index_path = snapshot_dir / "model.safetensors.index.json"
    for p in (config_path, tokenizer_path, model_file, index_path):
        if not p.exists():
            raise FileNotFoundError(p)

    config = json.load(open(config_path))
    text_cfg = dict(config["text_config"])
    audio_cfg = dict(config["audio_encoder_config"])
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    adapter = load_tokenizer(snapshot_dir)
    prompt_ids = [adapter.build_prompt(p) for p in prompts]
    max_len = max(len(x) for x in prompt_ids)
    pad_id = int(text_cfg.get("eos_token_id") or 151643)
    input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long, device=args.device)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long, device=args.device)
    for i, ids in enumerate(prompt_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
        attention_mask[i, : len(ids)] = 1
    prompt_lens = torch.tensor([len(x) for x in prompt_ids], dtype=torch.int64)

    backbone = load_backbone(model_file, text_cfg, args.device)
    modality_weight = load_modality_head_weight(model_file, args.device)
    with torch.inference_mode():
        out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        row_idx = torch.arange(len(prompts), device=args.device)
        last_hidden = hidden[row_idx, prompt_lens.to(args.device) - 1, :].contiguous()
        logits = F.linear(last_hidden, modality_weight).reshape(len(prompts), int(audio_cfg["num_codebooks"]), int(audio_cfg["vocab_size"]))
        logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1)
        top_vals, top_ids = torch.topk(logprobs, k=64, dim=-1)
        argmax_ids = torch.argmax(logits, dim=-1).to(torch.int64)

    tensors = {
        "prompt.input_ids_padded": input_ids.cpu().to(torch.int64),
        "prompt.attention_mask": attention_mask.cpu().to(torch.int64),
        "prompt.lengths": prompt_lens.cpu(),
        "final_hidden.bf16": last_hidden.cpu().to(torch.bfloat16),
        "audio_logits.f32": logits.cpu().to(torch.float32),
        "audio_top64.ids": top_ids.cpu().to(torch.int64),
        "audio_top64.logprobs.f32": top_vals.cpu().to(torch.float32),
        "audio_argmax.ids": argmax_ids.cpu().to(torch.int64),
    }
    metadata = {
        "fixture_kind": "higgs-one-step-audio-logits-golden",
        "schema_version": "1",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "reference": "SGLang-Omni Higgs prompt builder plus Transformers Qwen3 backbone plus SGLang fused modality head semantics",
        "sglang_omni_reference_files": "sglang_omni/models/higgs_tts/text_tokenizer.py;sglang_omni/models/higgs_tts/modeling.py;sglang_omni/models/higgs_tts/model.py",
        "prompt_count": str(len(prompts)),
        "prompts_json": json.dumps(prompts, ensure_ascii=False),
        "num_codebooks": str(audio_cfg["num_codebooks"]),
        "codebook_vocab_size": str(audio_cfg["vocab_size"]),
        "hidden_size": str(text_cfg["hidden_size"]),
        "config_sha256": sha256_file(config_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "model_index_sha256": sha256_file(index_path),
        "model_safetensors_size": str(model_file.stat().st_size),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "cuda_peak_allocated": str(torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0),
        "cuda_peak_reserved": str(torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"wrote {out_path} size={out_path.stat().st_size}")
    print("argmax", argmax_ids.cpu().tolist())


if __name__ == "__main__":
    main()
