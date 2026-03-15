#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/push_github_retry.sh <github_repo_url>"
  echo "Example: bash scripts/push_github_retry.sh https://github.com/1415302262/pharmacy-prediction-platform.git"
  exit 1
fi

REPO_URL="$1"
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
BRANCH="$(git branch --show-current)"
LOGFILE="push_github_retry.log"

if [ -z "$TOKEN" ]; then
  echo "ERROR: Please export GITHUB_TOKEN (or GH_TOKEN) in the current shell first." | tee -a "$LOGFILE"
  exit 1
fi

if [ -z "$BRANCH" ]; then
  echo "ERROR: Not inside a git repository." | tee -a "$LOGFILE"
  exit 1
fi

AUTHED_URL="$REPO_URL"
AUTHED_URL="${AUTHED_URL/https:\/\//https:\/\/x-access-token:${TOKEN}@}"

attempt=1
while true; do
  echo "[$(date '+%F %T')] Attempt ${attempt}: pushing ${BRANCH} to ${REPO_URL}" | tee -a "$LOGFILE"
  if git push "$AUTHED_URL" "$BRANCH":"$BRANCH" --set-upstream >> "$LOGFILE" 2>&1; then
    echo "[$(date '+%F %T')] Push succeeded." | tee -a "$LOGFILE"
    exit 0
  fi
  echo "[$(date '+%F %T')] Push failed. Retrying in 15 seconds..." | tee -a "$LOGFILE"
  attempt=$((attempt + 1))
  sleep 15
done
