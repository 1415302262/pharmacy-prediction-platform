#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/push_github.sh <GITHUB_REPO_URL>"
  echo "Example: bash scripts/push_github.sh https://github.com/1415302262/pharmacy-prediction-platform.git"
  exit 1
fi

REPO_URL="$1"
CURRENT_BRANCH="$(git branch --show-current)"

if [ -z "$CURRENT_BRANCH" ]; then
  echo "Not inside a git repository."
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git push -u origin "$CURRENT_BRANCH"
