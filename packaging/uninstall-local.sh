#!/bin/sh
set -eu
PREFIX=${YANLIU_PREFIX:-"$HOME/.local"}
case "$PREFIX" in /*) ;; *) echo 'YANLIU_PREFIX must be absolute' >&2; exit 1 ;; esac
case "${XDG_CONFIG_HOME:-}" in /*) CONFIG_DIR=$XDG_CONFIG_HOME ;; *) CONFIG_DIR="$HOME/.config" ;; esac
rm -f "$CONFIG_DIR/environment.d/60-yanliu.conf"
rm -f "$PREFIX/bin/yanliu-ibus" "$PREFIX/share/ibus/component/yanliu.xml"
rm -rf "$PREFIX/share/yanliu"
printf 'Removed Yanliu files from %s. Remove 砚流 in GNOME input sources if previously selected.\n' "$PREFIX"
