#!/usr/bin/env python3
"""Dump HF Higgs/Qwen3 layer-0 prefill stage snapshots for the one-step prompt."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from safetensors.torch import save_file

from dump_higgs_one_step_golden import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    DEFAULT_REVISION,
    load_backbone,
    load_tokenizer,
    sha256_file,
)


def last_token(hidden: torch.Tensor, row_idx: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    return hidden[row_idx, prompt_lens - 1, :].detach().contiguous().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    snapshot_dir = Path(args.snapshot_dir)
    config_path = snapshot_dir / "config.json"
    tokenizer_path = snapshot_dir / "tokenizer.json"
    model_file = snapshot_dir / "model.safetensors"
    index_path = snapshot_dir / "model.safetensors.index.json"
    for path in (config_path, tokenizer_path, model_file, index_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.load(open(config_path))
    text_cfg = dict(config["text_config"])
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    adapter = load_tokenizer(snapshot_dir)
    prompt_ids = [adapter.build_prompt(prompt) for prompt in prompts]
    max_len = max(len(ids) for ids in prompt_ids)
    pad_id = int(text_cfg.get("eos_token_id") or 151643)
    input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long, device=args.device)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long, device=args.device)
    for row, ids in enumerate(prompt_ids):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
        attention_mask[row, : len(ids)] = 1
    prompt_lens = torch.tensor([len(ids) for ids in prompt_ids], dtype=torch.int64)
    prompt_lens_device = prompt_lens.to(args.device)
    row_idx = torch.arange(len(prompt_ids), device=args.device)

    backbone = load_backbone(model_file, text_cfg, args.device)
    layer0 = backbone.layers[0]
    stages: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            stages[name] = last_token(tensor, row_idx, prompt_lens_device)

        return hook

    def capture_pre(name: str):
        def hook(_module, inputs):
            stages[name] = last_token(inputs[0], row_idx, prompt_lens_device)

        return hook

    hooks = [
        backbone.embed_tokens.register_forward_hook(capture("layer0.input_hidden.bf16")),
        layer0.input_layernorm.register_forward_hook(capture("layer0.input_norm.bf16")),
        layer0.self_attn.q_proj.register_forward_hook(capture("layer0.q_proj.bf16")),
        layer0.self_attn.k_proj.register_forward_hook(capture("layer0.k_proj.bf16")),
        layer0.self_attn.v_proj.register_forward_hook(capture("layer0.v_proj.bf16")),
        layer0.self_attn.o_proj.register_forward_pre_hook(capture_pre("layer0.attn_output.bf16")),
        layer0.self_attn.o_proj.register_forward_hook(capture("layer0.o_proj.bf16")),
        layer0.post_attention_layernorm.register_forward_hook(capture("layer0.post_attn_norm.bf16")),
        layer0.mlp.gate_proj.register_forward_hook(capture("layer0.gate_proj.bf16")),
        layer0.mlp.up_proj.register_forward_hook(capture("layer0.up_proj.bf16")),
        layer0.mlp.down_proj.register_forward_pre_hook(capture_pre("layer0.silu_mul.bf16")),
        layer0.mlp.down_proj.register_forward_hook(capture("layer0.down_proj.bf16")),
        layer0.register_forward_hook(capture("layer0.output_hidden.bf16")),
    ]
    try:
        with torch.inference_mode():
            backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    expected = [
        "layer0.input_hidden.bf16",
        "layer0.input_norm.bf16",
        "layer0.q_proj.bf16",
        "layer0.k_proj.bf16",
        "layer0.v_proj.bf16",
        "layer0.attn_output.bf16",
        "layer0.o_proj.bf16",
        "layer0.post_attn_norm.bf16",
        "layer0.gate_proj.bf16",
        "layer0.up_proj.bf16",
        "layer0.silu_mul.bf16",
        "layer0.down_proj.bf16",
        "layer0.output_hidden.bf16",
    ]
    missing = [name for name in expected if name not in stages]
    if missing:
        raise RuntimeError(f"missing stage snapshots: {missing}")

    tensors = {
        "prompt.input_ids_padded": input_ids.cpu().to(torch.int64),
        "prompt.attention_mask": attention_mask.cpu().to(torch.int64),
        "prompt.lengths": prompt_lens.cpu(),
    }
    for name in expected:
        tensors[name] = stages[name].cpu().to(torch.bfloat16)

    metadata = {
        "fixture_kind": "higgs-layer0-stage-golden",
        "schema_version": "1",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "reference": "Transformers Qwen3Model layer-0 module hooks",
        "prompt_count": str(len(prompts)),
        "prompts_json": json.dumps(prompts, ensure_ascii=False),
        "config_sha256": sha256_file(config_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "model_index_sha256": sha256_file(index_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    print(f"wrote {out_path} size={out_path.stat().st_size}")
    print(f"stages {len(expected)}")


if __name__ == "__main__":
    main()
