#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp_root="${TMPDIR:-/tmp}/pegainfer-higgs-isolated"

rm -rf "$tmp_root"
mkdir -p "$tmp_root"
cp -R "$repo_root/pegainfer-higgs-audio" "$tmp_root/pegainfer-higgs-audio"
python3 - <<'PY' "$tmp_root/pegainfer-higgs-audio/Cargo.toml"
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
text = re.sub(r'^pegainfer-core = .*\n', '', text, flags=re.MULTILINE)
text = re.sub(r'^pegainfer-qwen3-4b = .*\n', '', text, flags=re.MULTILINE)
text = re.sub(r'runtime-qwen3 = .*\n', 'runtime-qwen3 = []\n', text)
text = re.sub(r'\n\[\[bin\]\]\nname = "higgs_dump_one_step_actual"\npath = "src/bin/higgs_dump_one_step_actual.rs"\nrequired-features = \["runtime-qwen3"\]\n', '\n', text)
text = re.sub(r'\n\[\[bin\]\]\nname = "higgs_dump_prefill_layer_hidden"\npath = "src/bin/higgs_dump_prefill_layer_hidden.rs"\nrequired-features = \["runtime-qwen3"\]\n', '\n', text)
path.write_text(text)
PY
rm -f "$tmp_root/pegainfer-higgs-audio/src/bin/higgs_dump_one_step_actual.rs"
rm -f "$tmp_root/pegainfer-higgs-audio/src/bin/higgs_dump_prefill_layer_hidden.rs"
mkdir -p "$tmp_root/test_data"
cp "$repo_root/test_data/higgs-one-step-audio-logits.safetensors" "$tmp_root/test_data/"

cat >"$tmp_root/Cargo.toml" <<'TOML'
[workspace]
resolver = "3"
members = ["pegainfer-higgs-audio"]

[workspace.package]
version = "1.2.0"
edition = "2024"
license = "Apache-2.0"

[workspace.dependencies]
anyhow = "1.0"
clap = { version = "4.6.1", features = ["derive"] }
half = { version = "2.7", features = ["num-traits"] }
memmap2 = "0.9"
safetensors = "0.7"
serde_json = "1.0.149"
sha2 = "0.11"
tempfile = "3"

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -2 }
cast_lossless = "allow"
cast_possible_truncation = "allow"
cast_possible_wrap = "allow"
cast_precision_loss = "allow"
cast_sign_loss = "allow"
collapsible_else_if = "allow"
collapsible_if = "allow"
doc_markdown = "allow"
implicit_hasher = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
module_name_repetitions = "allow"
must_use_candidate = "allow"
similar_names = "allow"
too_many_arguments = "allow"
too_many_lines = "allow"
uninlined_format_args = "allow"
upper_case_acronyms = "allow"
redundant_clone = "warn"
unused_peekable = "warn"
dbg_macro = "warn"
exit = "warn"
get_unwrap = "warn"
print_stdout = "allow"
print_stderr = "allow"
rc_buffer = "warn"
rc_mutex = "warn"
rest_pat_in_fully_bound_structs = "warn"

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "warn"
unreachable_pub = "warn"
TOML

cd "$tmp_root"
cargo fmt --all --check
cargo test -p pegainfer-higgs-audio
cargo run -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --golden test_data/higgs-one-step-audio-logits.safetensors \
  --actual test_data/higgs-one-step-audio-logits.safetensors
