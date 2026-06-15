#!/usr/bin/env bash
# Build the paper with pandoc. Tries PDF (needs a LaTeX engine); falls back to HTML.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc not found. Install with:" >&2
  echo "  sudo apt-get install -y pandoc                # core" >&2
  echo "  sudo apt-get install -y texlive-xetex texlive-fonts-recommended  # for PDF" >&2
  exit 1
fi

COMMON=(paper.md --citeproc --bibliography refs.bib --metadata link-citations=true)

# Pick a LaTeX engine if available.
ENGINE=""
for e in tectonic xelatex pdflatex lualatex; do
  if command -v "$e" >/dev/null 2>&1; then ENGINE="$e"; break; fi
done

if [ -n "$ENGINE" ]; then
  echo "Building PDF with $ENGINE ..."
  pandoc "${COMMON[@]}" --pdf-engine="$ENGINE" -o paper.pdf
  echo "Wrote paper.pdf"
else
  echo "No LaTeX engine found; building self-contained HTML instead."
  pandoc "${COMMON[@]}" --standalone --embed-resources -o paper.html
  echo "Wrote paper.html"
  echo "For PDF: sudo apt-get install -y texlive-xetex && ./build.sh"
fi
