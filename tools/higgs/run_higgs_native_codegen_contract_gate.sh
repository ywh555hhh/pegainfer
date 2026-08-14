#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the Higgs Audio native code-generation contract gate.

This gate does not claim native audio decode. It validates the model-local
incremental audio-code contract and, optionally on a CUDA host, the runtime-qwen3
bridge tests that seed a retained prompt session.

Optional:
  --result-root DIR        Result directory.
                           Default: /data/results/pegainfer/higgs-audio
  --label LABEL            Output label. Default: current git short SHA.
  --sm SM                  PEGAINFER_CUDA_SM for CUDA checks. Default: 89
  --nvcc-jobs N            PEGAINFER_NVCC_JOBS for CUDA checks. Default: 8
  --runtime-qwen3          Also run runtime-qwen3 bridge tests.
  --model-dir DIR          Higgs checkpoint dir for the native continuation smoke.
  --golden FILE            Golden safetensors containing prompt tensors for the native continuation smoke.
  --qwen3-body-dir DIR     Optional Qwen3-compatible body view for runtime-qwen3 smoke.
  --qwen3-config-dir DIR   Optional Qwen3 config-only view for runtime-qwen3 smoke.
  --reference-trace FILE   Official/HF incremental reference trace JSON.
                           When set, compare it against the native continuation trace.
  --forced-prefix          Also run a forced common-prefix diagnostic using
                           sampled rows from --reference-trace while still
                           recording native retained-KV/audio-head logits.
  --native-loop-cmd CMD    Optional command to run under nsys/ncu. When omitted,
                           --profile uses the native continuation smoke if
                           --runtime-qwen3, --model-dir, and --golden are set.
  --profile                Capture nsys/ncu for the native continuation path.
  -h, --help               Show this help.

Outputs:
  <result-root>/native-codegen/higgs-continuation-contract-<label>.txt
  <result-root>/native-codegen/higgs-lib-tests-<label>.txt
  <result-root>/native-codegen/higgs-artifact-sampled-<label>.json
  <result-root>/native-codegen/higgs-artifact-trace-<label>.json
  <result-root>/native-codegen/higgs-artifact-codec-input-<label>.json
  <result-root>/native-codegen/higgs-artifact-tool-<label>.txt
  <result-root>/native-codegen/higgs-runtime-qwen3-tests-<label>.txt when --runtime-qwen3 is set
  <result-root>/native-codegen/higgs-native-continuation-smoke-<label>.txt when --runtime-qwen3, --model-dir, and --golden are set
  <result-root>/native-codegen/higgs-native-continuation-trace-<label>.json when the smoke runs
  <result-root>/native-codegen/higgs-native-continuation-codec-input-<label>.json when the smoke runs
  <result-root>/native-codegen/higgs-codegen-trace-compare-<label>.txt when --reference-trace is set
  <result-root>/native-codegen/higgs-native-forced-prefix-smoke-<label>.txt when --forced-prefix is set
  <result-root>/native-codegen/higgs-native-forced-prefix-trace-<label>.json when --forced-prefix is set
  <result-root>/native-codegen/higgs-forced-prefix-trace-compare-<label>.txt when --forced-prefix is set
  <result-root>/native-codegen/higgs-native-codegen-contract-gate-<label>.txt
  <result-root>/profiles/higgs-native-codegen-<label>.nsys-rep when --profile is set
  <result-root>/profiles/higgs-native-codegen-ncu-<label>.ncu-rep when --profile is set
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result_root="/data/results/pegainfer/higgs-audio"
label="$(git -C "$repo_root" rev-parse --short HEAD)"
sm="89"
nvcc_jobs="8"
runtime_qwen3=0
profile=0
forced_prefix=0
native_loop_cmd=""
model_dir=""
golden=""
qwen3_body_dir=""
qwen3_config_dir=""
reference_trace=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-root)
      result_root="${2:?missing value for --result-root}"
      shift 2
      ;;
    --label)
      label="${2:?missing value for --label}"
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
    --runtime-qwen3)
      runtime_qwen3=1
      shift
      ;;
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --golden)
      golden="${2:?missing value for --golden}"
      shift 2
      ;;
    --qwen3-body-dir)
      qwen3_body_dir="${2:?missing value for --qwen3-body-dir}"
      shift 2
      ;;
    --qwen3-config-dir)
      qwen3_config_dir="${2:?missing value for --qwen3-config-dir}"
      shift 2
      ;;
    --reference-trace)
      reference_trace="${2:?missing value for --reference-trace}"
      shift 2
      ;;
    --forced-prefix)
      forced_prefix=1
      shift
      ;;
    --native-loop-cmd)
      native_loop_cmd="${2:?missing value for --native-loop-cmd}"
      shift 2
      ;;
    --profile)
      profile=1
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

if [[ -n "$qwen3_body_dir" && -n "$qwen3_config_dir" ]]; then
  echo "--qwen3-body-dir and --qwen3-config-dir are mutually exclusive" >&2
  exit 2
fi
if [[ -n "$model_dir" || -n "$golden" || -n "$qwen3_body_dir" || -n "$qwen3_config_dir" ]]; then
  if [[ "$runtime_qwen3" -ne 1 ]]; then
    echo "--model-dir/--golden/--qwen3-* require --runtime-qwen3" >&2
    exit 2
  fi
fi
if [[ -n "$model_dir" && -z "$golden" || -z "$model_dir" && -n "$golden" ]]; then
  echo "--model-dir and --golden must be provided together" >&2
  exit 2
fi
if [[ -n "$reference_trace" ]]; then
  if [[ "$runtime_qwen3" -ne 1 || -z "$model_dir" || -z "$golden" ]]; then
    echo "--reference-trace requires --runtime-qwen3, --model-dir, and --golden" >&2
    exit 2
  fi
  if [[ ! -f "$reference_trace" ]]; then
    echo "reference trace not found: $reference_trace" >&2
    exit 1
  fi
fi
if [[ "$forced_prefix" -eq 1 && -z "$reference_trace" ]]; then
  echo "--forced-prefix requires --reference-trace" >&2
  exit 2
fi
if [[ "$profile" -eq 1 && -z "$native_loop_cmd" ]]; then
  if [[ "$runtime_qwen3" -ne 1 || -z "$model_dir" || -z "$golden" ]]; then
    echo "--profile without --native-loop-cmd requires --runtime-qwen3, --model-dir, and --golden" >&2
    exit 2
  fi
fi

require_nonempty_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "required output missing or empty: $path" >&2
    exit 1
  fi
}

capture_one_line() {
  local fallback="$1"
  shift
  local value
  if value="$("$@" 2>/dev/null | head -1)"; then
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return
    fi
  fi
  printf '%s' "$fallback"
}

capture_nvidia_smi() {
  local value
  if command -v nvidia-smi >/dev/null 2>&1; then
    if value="$(
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        | paste -sd ';' -
    )" && [[ -n "$value" ]]; then
      printf '%s' "$value"
      return
    fi
    if value="$(nvidia-smi -L 2>/dev/null | paste -sd ';' -)" && [[ -n "$value" ]]; then
      printf '%s' "$value"
      return
    fi
  fi
  printf '%s' "unavailable"
}

native_dir="$result_root/native-codegen"
profile_dir="$result_root/profiles"
contract_log="$native_dir/higgs-continuation-contract-$label.txt"
lib_test_log="$native_dir/higgs-lib-tests-$label.txt"
artifact_sampled_json="$native_dir/higgs-artifact-sampled-$label.json"
artifact_trace_json="$native_dir/higgs-artifact-trace-$label.json"
artifact_codec_json="$native_dir/higgs-artifact-codec-input-$label.json"
artifact_tool_log="$native_dir/higgs-artifact-tool-$label.txt"
runtime_qwen3_log="$native_dir/higgs-runtime-qwen3-tests-$label.txt"
native_continuation_log="$native_dir/higgs-native-continuation-smoke-$label.txt"
native_continuation_trace="$native_dir/higgs-native-continuation-trace-$label.json"
native_continuation_codec="$native_dir/higgs-native-continuation-codec-input-$label.json"
trace_compare_log="$native_dir/higgs-codegen-trace-compare-$label.txt"
forced_prefix_log="$native_dir/higgs-native-forced-prefix-smoke-$label.txt"
forced_prefix_trace="$native_dir/higgs-native-forced-prefix-trace-$label.json"
forced_prefix_codec="$native_dir/higgs-native-forced-prefix-codec-input-$label.json"
forced_prefix_compare_log="$native_dir/higgs-forced-prefix-trace-compare-$label.txt"
gate_summary="$native_dir/higgs-native-codegen-contract-gate-$label.txt"
nsys_report="$profile_dir/higgs-native-codegen-$label.nsys-rep"
ncu_report="$profile_dir/higgs-native-codegen-ncu-$label.ncu-rep"
nsys_stats_csv="$profile_dir/higgs-native-codegen-$label-stats_cuda_gpu_kern_sum.csv"
ncu_log="$profile_dir/higgs-native-codegen-ncu-$label.log"

mkdir -p "$native_dir" "$profile_dir"
cd "$repo_root"

export PEGAINFER_CUDA_SM="$sm"
export PEGAINFER_NVCC_JOBS="$nvcc_jobs"

gpu_info="$(capture_nvidia_smi)"
cuda_version="$(capture_one_line unavailable nvcc --version)"
nsys_version="$(capture_one_line unavailable nsys --version)"
ncu_version="$(capture_one_line unavailable ncu --version)"
rustc_version="$(capture_one_line unavailable rustc --version)"
cargo_version="$(capture_one_line unavailable cargo --version)"
uname_info="$(capture_one_line unavailable uname -a)"

echo "==> Higgs native code-generation contract gate"
echo "repo:          $repo_root"
echo "commit:        $(git -C "$repo_root" rev-parse --short HEAD)"
echo "label:         $label"
echo "summary:       $gate_summary"
echo "sm:            $PEGAINFER_CUDA_SM"
echo "nvcc_jobs:     $PEGAINFER_NVCC_JOBS"
echo "runtime_qwen3: $runtime_qwen3"
echo "gpu_info:      $gpu_info"
echo "rustc:         $rustc_version"

echo "==> Contract report"
cargo run --release -p pegainfer-higgs-audio --bin higgs_continuation_contract_report \
  2>&1 \
  | tee "$contract_log"
grep -q "input: feedback_embedding" "$contract_log"
grep -q "output: final_normed_hidden" "$contract_log"
grep -q "retained_kv: true" "$contract_log"
grep -q "full_prompt_rebuild: false" "$contract_log"
require_nonempty_file "$contract_log"

echo "==> Default Higgs lib tests"
cargo test --release -p pegainfer-higgs-audio --lib 2>&1 | tee "$lib_test_log"
grep -q "test result: ok" "$lib_test_log"
require_nonempty_file "$lib_test_log"

echo "==> Artifact schema smoke"
cat >"$artifact_sampled_json" <<'JSON'
{
  "sampled_codes": [
    [1, 101, 201, 301, 401, 501, 601, 701],
    [2, 102, 202, 302, 402, 502, 602, 702],
    [3, 103, 203, 303, 403, 503, 603, 703],
    [4, 104, 204, 304, 404, 504, 604, 704],
    [5, 105, 205, 305, 405, 505, 605, 705],
    [6, 106, 206, 306, 406, 506, 606, 706],
    [7, 107, 207, 307, 407, 507, 607, 707],
    [1025, 108, 208, 308, 408, 508, 608, 708]
  ]
}
JSON
cargo run --release -p pegainfer-higgs-audio --bin higgs_prepare_codegen_artifacts -- \
  --sampled-codes-json "$artifact_sampled_json" \
  --prompt-tokens 12 \
  --trace-out "$artifact_trace_json" \
  --codec-input-out "$artifact_codec_json" \
  2>&1 \
  | tee "$artifact_tool_log"
grep -q "higgs prepare codegen artifacts: ok" "$artifact_tool_log"
require_nonempty_file "$artifact_sampled_json"
require_nonempty_file "$artifact_trace_json"
require_nonempty_file "$artifact_codec_json"
require_nonempty_file "$artifact_tool_log"

runtime_qwen3_status="skipped"
native_continuation_status="skipped"
trace_compare_status="skipped"
forced_prefix_status="skipped"
forced_prefix_compare_status="skipped"
if [[ "$runtime_qwen3" -eq 1 ]]; then
  echo "==> runtime-qwen3 bridge tests"
  cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture \
    2>&1 \
    | tee "$runtime_qwen3_log"
  grep -q "test result: ok" "$runtime_qwen3_log"
  require_nonempty_file "$runtime_qwen3_log"
  runtime_qwen3_status="ok"

  if [[ -n "$model_dir" && -n "$golden" ]]; then
    echo "==> native retained continuation smoke"
    native_smoke_args=(
      --model-dir "$model_dir"
      --golden "$golden"
      --trace-out "$native_continuation_trace"
      --codec-input-out "$native_continuation_codec"
    )
    if [[ -n "$qwen3_body_dir" ]]; then
      native_smoke_args+=(--qwen3-body-dir "$qwen3_body_dir")
    fi
    if [[ -n "$qwen3_config_dir" ]]; then
      native_smoke_args+=(--qwen3-config-dir "$qwen3_config_dir")
    fi
    cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 --bin higgs_native_continuation_smoke -- "${native_smoke_args[@]}" \
      2>&1 \
      | tee "$native_continuation_log"
    grep -q "higgs native continuation smoke: ok" "$native_continuation_log"
    grep -q "continuation_steps: 8" "$native_continuation_log"
    grep -q "trace_steps: 9" "$native_continuation_log"
    require_nonempty_file "$native_continuation_log"
    require_nonempty_file "$native_continuation_trace"
    require_nonempty_file "$native_continuation_codec"
    native_continuation_status="ok"

    if [[ -n "$reference_trace" ]]; then
      echo "==> reference trace semantic comparison"
      if cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_codegen_trace -- \
        --reference "$reference_trace" \
        --actual "$native_continuation_trace" \
        2>&1 \
        | tee "$trace_compare_log"; then
        grep -q "higgs codegen trace comparison: ok" "$trace_compare_log"
        trace_compare_status="ok"
      else
        trace_compare_status="failed"
      fi
      require_nonempty_file "$trace_compare_log"

      if [[ "$forced_prefix" -eq 1 ]]; then
        echo "==> forced common-prefix retained continuation smoke"
        forced_prefix_args=(
          --model-dir "$model_dir"
          --golden "$golden"
          --reference-trace "$reference_trace"
          --trace-out "$forced_prefix_trace"
          --codec-input-out "$forced_prefix_codec"
        )
        if [[ -n "$qwen3_body_dir" ]]; then
          forced_prefix_args+=(--qwen3-body-dir "$qwen3_body_dir")
        fi
        if [[ -n "$qwen3_config_dir" ]]; then
          forced_prefix_args+=(--qwen3-config-dir "$qwen3_config_dir")
        fi
        if cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
          --bin higgs_native_forced_prefix_smoke -- "${forced_prefix_args[@]}" \
          2>&1 | tee "$forced_prefix_log"; then
          grep -q "higgs native forced-prefix smoke: ok" "$forced_prefix_log"
          grep -q "continuation_steps: 8" "$forced_prefix_log"
          grep -q "trace_steps: 9" "$forced_prefix_log"
          require_nonempty_file "$forced_prefix_trace"
          require_nonempty_file "$forced_prefix_codec"
          forced_prefix_status="ok"

          echo "==> forced common-prefix trace comparison"
          if cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_codegen_trace -- \
            --reference "$reference_trace" \
            --actual "$forced_prefix_trace" \
            2>&1 | tee "$forced_prefix_compare_log"; then
            grep -q "higgs codegen trace comparison: ok" "$forced_prefix_compare_log"
            forced_prefix_compare_status="ok"
          else
            forced_prefix_compare_status="failed"
          fi
          require_nonempty_file "$forced_prefix_compare_log"
        else
          forced_prefix_status="failed"
          require_nonempty_file "$forced_prefix_log"
        fi
      fi
    fi
  fi
fi

profile_status="skipped"
nsys_profile_status="skipped"
ncu_profile_status="skipped"
if [[ "$profile" -eq 1 ]]; then
  command -v nsys >/dev/null 2>&1 || { echo "nsys not found" >&2; exit 1; }
  command -v ncu >/dev/null 2>&1 || { echo "ncu not found" >&2; exit 1; }

  if [[ -z "$native_loop_cmd" ]]; then
    profile_trace="$native_dir/higgs-native-continuation-trace-$label-profiled.json"
    profile_codec="$native_dir/higgs-native-continuation-codec-input-$label-profiled.json"
    native_smoke_profile_args=(
      --model-dir "$model_dir"
      --golden "$golden"
      --trace-out "$profile_trace"
      --codec-input-out "$profile_codec"
    )
    if [[ -n "$qwen3_body_dir" ]]; then
      native_smoke_profile_args+=(--qwen3-body-dir "$qwen3_body_dir")
    fi
    if [[ -n "$qwen3_config_dir" ]]; then
      native_smoke_profile_args+=(--qwen3-config-dir "$qwen3_config_dir")
    fi
    printf -v native_loop_cmd "%q " \
      cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
      --bin higgs_native_continuation_smoke -- "${native_smoke_profile_args[@]}"
  fi

  echo "==> NSYS native loop profile"
  if nsys profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node \
    -o "${nsys_report%.nsys-rep}" --force-overwrite true bash -lc "$native_loop_cmd"; then
    require_nonempty_file "$nsys_report"
    nsys_profile_status="ok"
  else
    nsys_profile_status="failed"
  fi

  stats_base="$profile_dir/higgs-native-codegen-$label-stats"
  if [[ "$nsys_profile_status" == "ok" ]]; then
    if nsys stats --report cuda_gpu_kern_sum --format csv --output "$stats_base" \
      "$nsys_report"; then
      if [[ -f "$nsys_stats_csv" ]]; then
        echo "==> Top CUDA kernels from NSYS"
        head -20 "$nsys_stats_csv"
      fi
    else
      echo "nsys stats failed; raw report is still available at $nsys_report" >&2
    fi
  fi

  echo "==> NCU native loop profile"
  if ncu --set full --target-processes all -o "${ncu_report%.ncu-rep}" --force-overwrite bash -lc "$native_loop_cmd" \
    2>&1 | tee "$ncu_log"; then
    require_nonempty_file "$ncu_report"
    ncu_profile_status="ok"
  else
    require_nonempty_file "$ncu_log"
    if grep -q "ERR_NVGPUCTRPERM" "$ncu_log"; then
      ncu_profile_status="blocked:nvgpuctrperm"
    else
      ncu_profile_status="failed"
    fi
  fi

  if [[ "$nsys_profile_status" == "ok" && "$ncu_profile_status" == "ok" ]]; then
    profile_status="ok"
  elif [[ "$nsys_profile_status" == "ok" || "$ncu_profile_status" == "ok" ]]; then
    profile_status="partial"
  else
    profile_status="failed"
  fi
fi

cat >"$gate_summary" <<SUMMARY
status=ok
repo=$repo_root
commit=$(git -C "$repo_root" rev-parse --short HEAD)
label=$label
sm=$PEGAINFER_CUDA_SM
nvcc_jobs=$PEGAINFER_NVCC_JOBS
gpu_info=$gpu_info
cuda_version=$cuda_version
nsys_version=$nsys_version
ncu_version=$ncu_version
rustc_version=$rustc_version
cargo_version=$cargo_version
uname=$uname_info
contract_report=$contract_log
lib_test_log=$lib_test_log
runtime_qwen3=$runtime_qwen3_status
runtime_qwen3_log=$runtime_qwen3_log
native_continuation=$native_continuation_status
native_continuation_log=$native_continuation_log
native_continuation_trace=$native_continuation_trace
native_continuation_codec=$native_continuation_codec
reference_trace=$reference_trace
trace_compare=$trace_compare_status
trace_compare_log=$trace_compare_log
forced_prefix=$forced_prefix_status
forced_prefix_log=$forced_prefix_log
forced_prefix_trace=$forced_prefix_trace
forced_prefix_codec=$forced_prefix_codec
forced_prefix_compare=$forced_prefix_compare_status
forced_prefix_compare_log=$forced_prefix_compare_log
artifact=ok
artifact_sampled_json=$artifact_sampled_json
artifact_trace_json=$artifact_trace_json
artifact_codec_json=$artifact_codec_json
artifact_tool_log=$artifact_tool_log
profile=$profile_status
native_loop_cmd=$native_loop_cmd
nsys_profile=$nsys_profile_status
nsys_report=$nsys_report
nsys_stats_csv=$nsys_stats_csv
ncu_profile=$ncu_profile_status
ncu_report=$ncu_report
ncu_log=$ncu_log
claim_boundary=native_contract_only_no_native_decode
SUMMARY
require_nonempty_file "$gate_summary"

echo "==> Gate summary"
cat "$gate_summary"
python3 "$repo_root/tools/higgs/check_higgs_native_codegen_contract_summary.py" "$gate_summary" \
  --expected-label "$label" \
  --expected-sm "$PEGAINFER_CUDA_SM" \
  --expected-nvcc-jobs "$PEGAINFER_NVCC_JOBS" \
  --check-files
