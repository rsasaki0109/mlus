#!/usr/bin/env bash
set -euo pipefail
export CARGO_HOME=/workspace/.mlus-tools/cargo
export RUSTUP_HOME=/workspace/.mlus-tools/rustup
export PATH="$CARGO_HOME/bin:$PATH"
cd /workspace/mlus
if [ ! -f Cargo.toml ]; then
  echo 'MLus source is missing; restore the prepared source snapshot first.' >&2
  exit 1
fi
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
  curl --fail --location --proto '=https' --tlsv1.2 https://sh.rustup.rs -o /tmp/mlus-rustup.sh
  sh /tmp/mlus-rustup.sh -y --no-modify-path --profile minimal --default-toolchain 1.90.0
fi
cargo build --locked
cargo build --release --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
python3 scripts/smoke.py
python3 scripts/profile_smoke.py
python3 scripts/checkpoint_smoke.py
python3 scripts/recovery_smoke.py
python3 scripts/process_smoke.py
python3 scripts/test_pytorch_wrapper.py
cargo run --locked --example sim_benchmark
cargo run --locked --example feedback_benchmark
