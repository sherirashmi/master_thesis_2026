#!/bin/sh
# Compiles the block diagrams: vector PDF (LaTeX fonts) and a 300 dpi PNG next to this folder's parent (thesis/figures).
cd "$(dirname "$0")"
for f in block_operators_overview block_forward_architecture block_idon_architecture block_invertible_operator_architecture; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$f.tex" >/dev/null || { echo "failed: $f"; exit 1; }
  cp "build/$f.pdf" ../
  pdftoppm -png -r 300 -singlefile "build/$f.pdf" "../$f"
  echo "built $f"
done
