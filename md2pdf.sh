#!/bin/bash

set -e

show_help() {
  echo "Usage: $0 <markdown-file>"
  echo
  echo "Converts a Markdown file to PDF."
  echo "The PDF is created in the same directory with the same base filename."
  echo
  echo "Options:"
  echo "  -h, --help    Show this help message"
  echo
  echo "Example:"
  echo "  $0 docs/Google-Classroom-Integration.md"
  echo
  echo "Output:"
  echo "  docs/Google-Classroom-Integration.pdf"
}

if [ "$#" -eq 0 ]; then
  show_help
  exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  show_help
  exit 0
fi

if [ "$#" -ne 1 ]; then
  echo "Error: Expected exactly one Markdown file."
  echo
  show_help
  exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: File not found: $INPUT_FILE"
  exit 1
fi

if [[ "$INPUT_FILE" != *.md ]]; then
  echo "Error: Input file must have a .md extension."
  exit 1
fi

OUTPUT_FILE="${INPUT_FILE%.md}.pdf"

pandoc "$INPUT_FILE" \
  -o "$OUTPUT_FILE" \
  --pdf-engine=xelatex \
  -V geometry:margin=0.8in

echo "PDF created: $OUTPUT_FILE"