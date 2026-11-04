#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v tectonic >/dev/null 2>&1; then
  tectonic seg301_ieee_paper.tex --keep-logs
  tectonic supplement.tex --keep-logs
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -halt-on-error seg301_ieee_paper.tex
  bibtex seg301_ieee_paper
  pdflatex -halt-on-error seg301_ieee_paper.tex
  pdflatex -halt-on-error seg301_ieee_paper.tex
  pdflatex -halt-on-error supplement.tex
  pdflatex -halt-on-error supplement.tex
else
  echo 'Install TeX Live/MiKTeX or Tectonic to compile.' >&2
  exit 1
fi
