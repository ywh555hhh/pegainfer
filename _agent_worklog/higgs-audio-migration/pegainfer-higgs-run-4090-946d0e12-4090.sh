#!/usr/bin/env bash
set -euo pipefail

# Run from the remote PegaInfer repository root after extracting the overlay:
#   tar -xzf /tmp/pegainfer-higgs-native-codegen-overlay-<label>.tgz -C /data/src/pegainfer

MODEL_DIR="${MODEL_DIR:-/path/to/higgs}"
GOLDEN="${GOLDEN:-/path/to/higgs-one-step-golden.safetensors}"
RESULT_ROOT="${RESULT_ROOT:-/data/results/pegainfer/higgs-audio}"
LABEL="${LABEL:-$(git rev-parse --short HEAD)-4090}"

export PEGAINFER_CUDA_SM="${PEGAINFER_CUDA_SM:-89}"
export PEGAINFER_NVCC_JOBS="${PEGAINFER_NVCC_JOBS:-8}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "MODEL_DIR not found: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$GOLDEN" ]]; then
  echo "GOLDEN not found: $GOLDEN" >&2
  exit 1
fi

if ! command -v protoc >/dev/null 2>&1; then
  apt-get update
  apt-get install -y protobuf-compiler
fi

rustc --version
cargo --version
nvidia-smi
nvcc --version || true
nsys --version || true
ncu --version || true

cargo check --release -p pegainfer-server --features higgs-audio
cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture

tools/higgs/run_higgs_4090_native_codegen_validation.sh \
  --result-root "$RESULT_ROOT" \
  --label "$LABEL" \
  --model-dir "$MODEL_DIR" \
  --golden "$GOLDEN" \
  --generate-reference-trace \
  --profile \
  --sm "$PEGAINFER_CUDA_SM" \
  --nvcc-jobs "$PEGAINFER_NVCC_JOBS"

python3 tools/higgs/check_higgs_native_codegen_contract_summary.py \
  "$RESULT_ROOT/native-codegen/higgs-native-codegen-contract-gate-$LABEL.txt" \
  --expected-label "$LABEL" \
  --expected-sm "$PEGAINFER_CUDA_SM" \
  --expected-nvcc-jobs "$PEGAINFER_NVCC_JOBS" \
  --check-files

python3 tools/higgs/check_higgs_4090_evidence_bundle.py \
  "$RESULT_ROOT/native-codegen/higgs-native-codegen-contract-gate-$LABEL.txt" \
  --pr-evidence "$RESULT_ROOT/native-codegen/pr-evidence-$LABEL.md" \
  --expected-label "$LABEL" \
  --expected-sm "$PEGAINFER_CUDA_SM" \
  --expected-nvcc-jobs "$PEGAINFER_NVCC_JOBS"

printf '\nPR evidence:\n%s\n' "$RESULT_ROOT/native-codegen/pr-evidence-$LABEL.md"
