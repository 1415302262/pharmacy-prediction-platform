#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash scripts/push_hf_space.sh <SPACE_REPO_URL> <LOCAL_SPACE_CLONE_DIR>"
  exit 1
fi

SPACE_REPO_URL="$1"
LOCAL_SPACE_CLONE_DIR="$2"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_SPACE_DIR="$PROJECT_ROOT/hf_space"

if [ ! -d "$HF_SPACE_DIR" ]; then
  echo "hf_space/ not found"
  exit 1
fi

if [ ! -d "$LOCAL_SPACE_CLONE_DIR/.git" ]; then
  git clone "$SPACE_REPO_URL" "$LOCAL_SPACE_CLONE_DIR"
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$HF_SPACE_DIR/" "$LOCAL_SPACE_CLONE_DIR/"
else
  rm -rf "$LOCAL_SPACE_CLONE_DIR"/*
  cp -a "$HF_SPACE_DIR"/. "$LOCAL_SPACE_CLONE_DIR/"
fi

cd "$LOCAL_SPACE_CLONE_DIR"
git add .
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "Deploy Flask-based lipophilicity demo"
git push
