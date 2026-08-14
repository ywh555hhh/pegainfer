#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the Higgs Audio 4090 native code-generation validation sequence.

This is a Linux/4090 driver around the lower-level native codegen contract gate.
It does not claim native wav E2E or production serving. It records the server
feature build check, native retained-continuation gate summary, optional
reference-trace comparison, optional profiling, and PR-ready evidence Markdown.

Required:
  --model-dir DIR          Higgs checkpoint directory.
  --golden FILE            Golden safetensors containing prompt tensors.

Optional:
  --result-root DIR        Result directory.
                           Default: /data/results/pegainfer/higgs-audio
  --label LABEL            Output label. Default: current git short SHA.
  --reference-trace FILE   Official/HF incremental reference trace JSON.
  --generate-reference-trace
                           Generate an HF incremental reference trace first.
  --forced-prefix          Also run forced common-prefix diagnostics when a
                           reference trace is available.
  --qwen3-body-dir DIR     Optional Qwen3-compatible body view for runtime-qwen3 smoke.
  --qwen3-config-dir DIR   Optional Qwen3 config-only view for runtime-qwen3 smoke.
  --prompt TEXT            Prompt for generated reference trace.
                           Default: Hello from PegaInfer.
  --reference-device DEV   Torch device for generated reference trace.
                           Default: cuda:0
  --reference-steps N      Audio continuation steps for generated reference trace.
                           Default: 8
  --sglang-omni-src DIR    Optional SGLang-Omni source tree for reference semantics.
  --profile                Capture nsys/ncu for the native continuation smoke.
  --sm SM                  PEGAINFER_CUDA_SM. Default: 89
  --nvcc-jobs N            PEGAINFER_NVCC_JOBS. Default: 8
  --install-protoc         Install protobuf-compiler with apt-get when protoc is missing.
  -h, --help               Show this help.

Outputs:
  <result-root>/native-codegen/higgs-server-check-<label>.txt
  <result-root>/native-codegen/higgs-native-codegen-contract-gate-<label>.txt
  <result-root>/native-codegen/pr-evidence-<label>.md
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result_root="/data/results/pegainfer/higgs-audio"
label="$(git -C "$repo_root" rev-parse --short HEAD)"
model_dir=""
golden=""
reference_trace=""
generate_reference_trace=0
forced_prefix=0
qwen3_body_dir=""
qwen3_config_dir=""
profile=0
sm="89"
nvcc_jobs="8"
install_protoc=0
prompt="Hello from PegaInfer."
reference_device="cuda:0"
reference_steps="8"
sglang_omni_src=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --golden)
      golden="${2:?missing value for --golden}"
      shift 2
      ;;
    --result-root)
      result_root="${2:?missing value for --result-root}"
      shift 2
      ;;
    --label)
      label="${2:?missing value for --label}"
      shift 2
      ;;
    --reference-trace)
      reference_trace="${2:?missing value for --reference-trace}"
      shift 2
      ;;
    --generate-reference-trace)
      generate_reference_trace=1
      shift
      ;;
    --forced-prefix)
      forced_prefix=1
      shift
      ;;
    --qwen3-body-dir)
      qwen3_body_dir="${2:?missing value for --qwen3-body-dir}"
      shift 2
      ;;
    --qwen3-config-dir)
      qwen3_config_dir="${2:?missing value for --qwen3-config-dir}"
      shift 2
      ;;
    --profile)
      profile=1
      shift
      ;;
    --prompt)
      prompt="${2:?missing value for --prompt}"
      shift 2
      ;;
    --reference-device)
      reference_device="${2:?missing value for --reference-device}"
      shift 2
      ;;
    --reference-steps)
      reference_steps="${2:?missing value for --reference-steps}"
      shift 2
      ;;
    --sglang-omni-src)
      sglang_omni_src="${2:?missing value for --sglang-omni-src}"
      shift 2
      ;;
    --sm)
      sm="${2:?missing value for --sm}"
      shift 2
      ;;
    --nvcc-jobs)
      nvcc_jobs="${2:?missing value for --nvcc-jobs}"
      shift 2
      ;;
    --install-protoc)
      install_protoc=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$model_dir" || -z "$golden" ]]; then
  echo "--model-dir and --golden are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$model_dir" ]]; then
  echo "model dir not found: $model_dir" >&2
  exit 1
fi
if [[ ! -f "$golden" ]]; then
  echo "golden file not found: $golden" >&2
  exit 1
fi
if [[ -n "$reference_trace" && ! -f "$reference_trace" ]]; then
  echo "reference trace not found: $reference_trace" >&2
  exit 1
fi
if [[ "$generate_reference_trace" -eq 1 && -n "$reference_trace" ]]; then
  echo "--generate-reference-trace and --reference-trace are mutually exclusive" >&2
  exit 2
fi
if [[ "$forced_prefix" -eq 1 && "$generate_reference_trace" -ne 1 && -z "$reference_trace" ]]; then
  echo "--forced-prefix requires --reference-trace or --generate-reference-trace" >&2
  exit 2
fi
if [[ -n "$qwen3_body_dir" && -n "$qwen3_config_dir" ]]; then
  echo "--qwen3-body-dir and --qwen3-config-dir are mutually exclusive" >&2
  exit 2
fi

require_nonempty_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "required output missing or empty: $path" >&2
    exit 1
  fi
}

if ! command -v protoc >/dev/null 2>&1; then
  if [[ "$install_protoc" -eq 1 ]]; then
    apt-get update
    apt-get install -y protobuf-compiler
  else
    echo "protoc not found; rerun with --install-protoc or install protobuf-compiler" >&2
    exit 1
  fi
fi

native_dir="$result_root/native-codegen"
mkdir -p "$native_dir"

server_check_log="$native_dir/higgs-server-check-$label.txt"
generated_reference_trace="$native_dir/higgs-reference-incremental-trace-$label.json"
generated_reference_codec="$native_dir/higgs-reference-incremental-codec-input-$label.json"
gate_summary="$native_dir/higgs-native-codegen-contract-gate-$label.txt"
pr_evidence="$native_dir/pr-evidence-$label.md"

cd "$repo_root"
export PEGAINFER_CUDA_SM="$sm"
export PEGAINFER_NVCC_JOBS="$nvcc_jobs"

echo "==> Higgs 4090 native codegen validation"
echo "repo:       $repo_root"
echo "commit:     $(git rev-parse --short HEAD)"
echo "label:      $label"
echo "model_dir:  $model_dir"
echo "golden:     $golden"
echo "result:     $result_root"
echo "profile:    $profile"
echo "forced_prefix: $forced_prefix"

if [[ "$generate_reference_trace" -eq 1 ]]; then
  echo "==> Official/HF incremental reference trace"
  reference_args=(
    --snapshot-dir "$model_dir"
    --prompt "$prompt"
    --device "$reference_device"
    --steps "$reference_steps"
    --out "$generated_reference_trace"
    --codec-input-out "$generated_reference_codec"
  )
  if [[ -n "$sglang_omni_src" ]]; then
    reference_args+=(--sglang-omni-src "$sglang_omni_src")
  fi
  python3 tools/higgs/dump_higgs_incremental_trace_reference.py "${reference_args[@]}"
  reference_trace="$generated_reference_trace"
fi

if [[ -n "$reference_trace" ]]; then
  echo "==> Reference trace schema/provenance check"
  python3 tools/higgs/check_higgs_incremental_trace_reference.py "$reference_trace" \
    --expected-steps "$((reference_steps + 1))"
fi

echo "==> Server feature check"
cargo check --release -p pegainfer-server --features higgs-audio \
  2>&1 \
  | tee "$server_check_log"
require_nonempty_file "$server_check_log"

gate_args=(
  --result-root "$result_root"
  --label "$label"
  --runtime-qwen3
  --model-dir "$model_dir"
  --golden "$golden"
  --sm "$sm"
  --nvcc-jobs "$nvcc_jobs"
)
if [[ -n "$reference_trace" ]]; then
  gate_args+=(--reference-trace "$reference_trace")
fi
if [[ "$forced_prefix" -eq 1 ]]; then
  gate_args+=(--forced-prefix)
fi
if [[ -n "$qwen3_body_dir" ]]; then
  gate_args+=(--qwen3-body-dir "$qwen3_body_dir")
fi
if [[ -n "$qwen3_config_dir" ]]; then
  gate_args+=(--qwen3-config-dir "$qwen3_config_dir")
fi
if [[ "$profile" -eq 1 ]]; then
  gate_args+=(--profile)
fi

echo "==> Native codegen contract gate"
tools/higgs/run_higgs_native_codegen_contract_gate.sh "${gate_args[@]}"

echo "==> PR evidence Markdown"
python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py "$gate_summary" \
  --out "$pr_evidence"
require_nonempty_file "$pr_evidence"

echo "==> 4090 evidence bundle check"
python3 tools/higgs/check_higgs_4090_evidence_bundle.py "$gate_summary" \
  --pr-evidence "$pr_evidence" \
  --expected-label "$label" \
  --expected-sm "$sm" \
  --expected-nvcc-jobs "$nvcc_jobs"

echo "==> Higgs 4090 native codegen validation: ok"
echo "server_check=$server_check_log"
echo "gate_summary=$gate_summary"
echo "pr_evidence=$pr_evidence"
