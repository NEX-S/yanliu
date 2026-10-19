#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
case "${YANLIU_DEVICE:-cpu}" in
    cpu) FEATURES=ibus ;;
    gpu|npu) FEATURES=ibus,accelerator ;;
    *) echo 'YANLIU_DEVICE must be cpu, gpu or npu' >&2; exit 1 ;;
esac
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-cpu=native" cargo build --locked --release --features "$FEATURES" --bin yanliu-ibus
python3 packaging/install_assets.py
