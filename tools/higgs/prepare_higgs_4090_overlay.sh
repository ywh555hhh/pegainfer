#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Prepare a Higgs Audio native-codegen overlay for a remote RTX 4090 host.

This helper packages the current worktree changes needed for the Higgs-Audio
native-codegen validation slice while excluding dirty third-party kernel
submodules. It also writes a small remote runner template that executes the
4090 validation gate after the overlay is extracted into a PegaInfer checkout.

Optional:
  --out-dir DIR        Output directory. Default: /tmp
  --label LABEL        Artifact label. Default: current git short SHA
  --repo-root DIR      Repository root. Default: auto-detected
  -h, --help           Show this help.

Outputs:
  <out-dir>/pegainfer-higgs-native-codegen-overlay-<label>.tgz
  <out-dir>/pegainfer-higgs-native-codegen-overlay-files-<label>.txt
  <out-dir>/pegainfer-higgs-run-4090-<label>.sh
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out_dir="/tmp"
label="$(git -C "$repo_root" rev-parse --short HEAD)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --label)
      label="${2:?missing value for --label}"
      shift 2
      ;;
    --repo-root)
      repo_root="${2:?missing value for --repo-root}"
      shift 2
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

repo_root="$(cd "$repo_root" && pwd)"
mkdir -p "$out_dir"

overlay="$out_dir/pegainfer-higgs-native-codegen-overlay-$label.tgz"
file_list="$out_dir/pegainfer-higgs-native-codegen-overlay-files-$label.txt"
runner="$out_dir/pegainfer-higgs-run-4090-$label.sh"

changed_files=()
while IFS= read -r -d '' path; do
  case "$path" in
    pegainfer-kernels/third_party/*) continue ;;
  esac
  changed_files+=("$path")
done < <(
  {
    git -C "$repo_root" diff --name-only -z HEAD
    git -C "$repo_root" ls-files --others --exclude-standard -z
  } | sort -zu
)

if [[ "${#changed_files[@]}" -eq 0 ]]; then
  echo "no changed or untracked files to package" >&2
  exit 1
fi

printf '%s\n' "${changed_files[@]}" >"$file_list"
(
  cd "$repo_root"
  LC_ALL=C tar -czf "$overlay" "${changed_files[@]}"
)

cat >"$runner" <<'RUNNER'
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
RUNNER
chmod +x "$runner"

echo "higgs 4090 overlay: ok"
echo "  repo:      $repo_root"
echo "  label:     $label"
echo "  files:     ${#changed_files[@]}"
echo "  file_list: $file_list"
echo "  overlay:   $overlay"
echo "  runner:    $runner"
