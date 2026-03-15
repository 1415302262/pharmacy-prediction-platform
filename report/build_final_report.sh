#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/report"
INPUT_MD="$REPORT_DIR/final_complete_report_zh.md"
OUTPUT_TEX="$REPORT_DIR/final_complete_report_zh.tex"
OUTPUT_HTML="$REPORT_DIR/final_complete_report_zh.html"
OUTPUT_PDF="$REPORT_DIR/final_complete_report_zh.pdf"

pandoc "$INPUT_MD" -s -o "$OUTPUT_TEX" -V documentclass=ctexart
pandoc "$INPUT_MD" -s -o "$OUTPUT_HTML"

echo "Generated:"
echo "- $OUTPUT_TEX"
echo "- $OUTPUT_HTML"

if command -v xelatex >/dev/null 2>&1; then
  (cd "$REPORT_DIR" && xelatex -interaction=nonstopmode "$(basename "$OUTPUT_TEX")")
  echo "- $OUTPUT_PDF"
else
  echo "PDF not generated: xelatex not found. You can open the HTML file in a browser and print it to PDF."
fi
