#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.cargo
cat >/root/.cargo/config.toml <<'CARGO_CONFIG'
[source.crates-io]
replace-with = "ustc"

[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
CARGO_CONFIG

export RUSTUP_TOOLCHAIN=nightly
export PATH=/root/.cargo/bin:/usr/local/cuda-13.0/bin:/root/autodl-tmp/venvs/higgs-omni/bin:$PATH
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8
export CUDA_HOME=/usr/local/cuda-13.0
export LC_ALL=C
export LANG=C

cd /data/src/pegainfer
cargo check --release -p pegainfer-server --features higgs-audio
