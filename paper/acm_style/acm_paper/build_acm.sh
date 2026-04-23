#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACM_STYLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export TEXMFHOME="$ACM_STYLE_DIR/texmf"
export TEXMFVAR="$ACM_STYLE_DIR/texmf-var"
export TEXMFCONFIG="$ACM_STYLE_DIR/texmf-config"
export TEXINPUTS="../acm_template//:"

cd "$SCRIPT_DIR"

updmap-user --enable Map=libertine.map --enable Map=zi4.map --enable Map=newtx.map >/dev/null
pdflatex -interaction=nonstopmode robustsep_acm.tex
pdflatex -interaction=nonstopmode robustsep_acm.tex
