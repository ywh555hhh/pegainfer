#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run a Higgs Audio end-to-end wav gate.

This gate targets a real audio artifact. It first requests a reference wav from
an OpenAI-compatible /v1/audio/speech service. If --codes-json is provided, it
also replays delayed/raw Higgs code rows through the local PegaInfer Higgs codec
sidecar and writes a second wav from the same result directory.

Required:
  --input TEXT             Text prompt to synthesize.

Optional:
  --api-base URL           OpenAI-compatible API base. Default: http://localhost:8000
  --model MODEL            Request model id. Default: bosonai/higgs-audio-v3-tts-4b
  --voice VOICE            Request voice. Default: default
  --python PYTHON          Python executable. Default: python3
  --result-root DIR        Result directory. Default: /data/results/pegainfer/higgs-audio
  --label LABEL            Output label. Default: current git short SHA.
  --references-json FILE   JSON array for OpenAI-compatible references.
  --temperature FLOAT      Optional sampling temperature.
  --top-p FLOAT            Optional top-p.
  --top-k INT              Optional top-k.
  --max-new-tokens INT     Optional max_new_tokens.
  --seed INT               Optional seed.
  --codes-json FILE        Optional delayed/raw code rows to vocode through PegaInfer.
  --codes-layout LAYOUT    delayed or raw. Default: delayed.
  --model-dir DIR          Higgs checkpoint dir for codec sidecar; required with --codes-json.
  --device DEVICE          Torch device for codec sidecar. Default: cuda:0
  -h, --help               Show this help.

Outputs:
  <result-root>/audio/higgs-reference-e2e-<label>.wav
  <result-root>/audio/higgs-reference-e2e-<label>-payload.json
  <result-root>/audio/higgs-reference-e2e-<label>-summary.txt
  <result-root>/audio/higgs-codec-replay-<label>.wav when --codes-json is set
  <result-root>/audio/higgs-audio-e2e-gate-<label>.txt
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
api_base="http://localhost:8000"
model="bosonai/higgs-audio-v3-tts-4b"
voice="default"
python_bin="python3"
result_root="/data/results/pegainfer/higgs-audio"
label="$(git -C "$repo_root" rev-parse --short HEAD)"
input_text=""
references_json=""
temperature=""
top_p=""
top_k=""
max_new_tokens=""
seed=""
codes_json=""
codes_layout="delayed"
model_dir=""
device="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-base)
      api_base="${2:?missing value for --api-base}"
      shift 2
      ;;
    --model)
      model="${2:?missing value for --model}"
      shift 2
      ;;
    --voice)
      voice="${2:?missing value for --voice}"
      shift 2
      ;;
    --python)
      python_bin="${2:?missing value for --python}"
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
    --input)
      input_text="${2:?missing value for --input}"
      shift 2
      ;;
    --references-json)
      references_json="${2:?missing value for --references-json}"
      shift 2
      ;;
    --temperature)
      temperature="${2:?missing value for --temperature}"
      shift 2
      ;;
    --top-p)
      top_p="${2:?missing value for --top-p}"
      shift 2
      ;;
    --top-k)
      top_k="${2:?missing value for --top-k}"
      shift 2
      ;;
    --max-new-tokens)
      max_new_tokens="${2:?missing value for --max-new-tokens}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing value for --seed}"
      shift 2
      ;;
    --codes-json)
      codes_json="${2:?missing value for --codes-json}"
      shift 2
      ;;
    --codes-layout)
      codes_layout="${2:?missing value for --codes-layout}"
      shift 2
      ;;
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --device)
      device="${2:?missing value for --device}"
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

if [[ -z "$input_text" ]]; then
  echo "--input is required" >&2
  usage >&2
  exit 2
fi

case "$codes_layout" in
  delayed | raw) ;;
  *)
    echo "--codes-layout must be delayed or raw" >&2
    exit 2
    ;;
esac

if [[ -n "$codes_json" && -z "$model_dir" ]]; then
  echo "--model-dir is required when --codes-json is set" >&2
  exit 2
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python executable not found: $python_bin" >&2
  exit 1
fi

require_nonempty_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "required output missing or empty: $path" >&2
    exit 1
  fi
}

audio_dir="$result_root/audio"
reference_wav="$audio_dir/higgs-reference-e2e-$label.wav"
reference_summary="$audio_dir/higgs-reference-e2e-$label-summary.txt"
reference_payload="$audio_dir/higgs-reference-e2e-$label-payload.json"
gate_summary="$audio_dir/higgs-audio-e2e-gate-$label.txt"
codec_wav="$audio_dir/higgs-codec-replay-$label.wav"
codec_input="$audio_dir/higgs-codec-replay-$label-input.json"

mkdir -p "$audio_dir"
cd "$repo_root"

echo "==> Higgs Audio e2e wav gate"
echo "repo:          $repo_root"
echo "commit:        $(git -C "$repo_root" rev-parse --short HEAD)"
echo "api_base:      $api_base"
echo "model:         $model"
echo "label:         $label"
echo "reference_wav: $reference_wav"

"$python_bin" -m py_compile tools/higgs/request_higgs_audio.py

reference_args=(
  tools/higgs/request_higgs_audio.py
  --api-base "$api_base" \
  --model "$model" \
  --voice "$voice" \
  --input "$input_text" \
  --out-wav "$reference_wav" \
  --summary-out "$reference_summary" \
  --payload-out "$reference_payload"
)
[[ -n "$references_json" ]] && reference_args+=(--references-json "$references_json")
[[ -n "$temperature" ]] && reference_args+=(--temperature "$temperature")
[[ -n "$top_p" ]] && reference_args+=(--top-p "$top_p")
[[ -n "$top_k" ]] && reference_args+=(--top-k "$top_k")
[[ -n "$max_new_tokens" ]] && reference_args+=(--max-new-tokens "$max_new_tokens")
[[ -n "$seed" ]] && reference_args+=(--seed "$seed")
"$python_bin" "${reference_args[@]}"
require_nonempty_file "$reference_wav"
require_nonempty_file "$reference_summary"
require_nonempty_file "$reference_payload"

codec_replay="skipped"
codec_frames=""
if [[ -n "$codes_json" ]]; then
  echo "==> Higgs codec replay wav"
  codec_args=(
    run --release -p pegainfer-higgs-audio --bin higgs_vocode_codes --
    --model-dir "$model_dir" \
    --codes-json "$codes_json" \
    --codes-layout "$codes_layout" \
    --out-wav "$codec_wav" \
    --codec-input-out "$codec_input" \
    --python "$python_bin" \
    --device "$device"
  )
  cargo "${codec_args[@]}" | tee "$audio_dir/higgs-codec-replay-$label.log"
  require_nonempty_file "$codec_wav"
  require_nonempty_file "$codec_input"
  codec_replay="ok"
  codec_frames="$("$python_bin" - <<'PY' "$codec_input"
import json
import sys
from pathlib import Path
print(len(json.loads(Path(sys.argv[1]).read_text())["raw_codes"]))
PY
)"
fi

duration_sec="$(grep '^duration_sec=' "$reference_summary" | cut -d= -f2-)"
sample_rate="$(grep '^sample_rate=' "$reference_summary" | cut -d= -f2-)"
peak_abs="$(grep '^peak_abs=' "$reference_summary" | cut -d= -f2-)"

cat >"$gate_summary" <<SUMMARY
status=ok
repo=$repo_root
commit=$(git -C "$repo_root" rev-parse --short HEAD)
label=$label
api_base=$api_base
model=$model
voice=$voice
reference_wav=$reference_wav
reference_summary=$reference_summary
reference_payload=$reference_payload
sample_rate=$sample_rate
duration_sec=$duration_sec
peak_abs=$peak_abs
codec_replay=$codec_replay
codec_wav=$codec_wav
codec_input=$codec_input
codec_frames=$codec_frames
SUMMARY
require_nonempty_file "$gate_summary"

echo "==> Gate summary"
cat "$gate_summary"
echo "==> Higgs Audio e2e wav gate: ok"
