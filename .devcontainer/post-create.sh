#!/usr/bin/env bash
set -euo pipefail

py_version="$(
  python - <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    print("", end="")
    sys.exit(0)

pyproject = Path("pyproject.toml")
if not pyproject.exists():
    print("", end="")
    sys.exit(0)

data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
req = (data.get("project") or {}).get("requires-python") or ""

version = ""
for pattern in (r"==\s*([0-9]+(?:\.[0-9]+){1,2})", r">=\s*([0-9]+(?:\.[0-9]+){1,2})"):
    match = re.search(pattern, req)
    if match:
        version = match.group(1)
        break

print(version, end="")
PY
)"

if [[ -n "${py_version}" ]]; then
  echo "Using Python ${py_version} from pyproject.toml requires-python"
  uv python install "${py_version}"
  uv venv --python "${py_version}"
else
  echo "No requires-python found; using default interpreter"
  uv venv
fi

uv sync --all-groups

# Configure git if not already configured
if ! git config --global user.name > /dev/null 2>&1; then
  echo "Configuring git user settings..."
  # Use GitHub-provided user info if available (Codespaces)
  if [[ -n "${GITHUB_USER:-}" ]]; then
    git config --global user.name "${GITHUB_USER}"
    git config --global user.email "${GITHUB_USER}@users.noreply.github.com"
  # Try GitHub CLI if authenticated
  elif command -v gh > /dev/null 2>&1 && gh auth status > /dev/null 2>&1; then
    GITHUB_USER_NAME=$(gh api user --jq '.name // .login')
    GITHUB_USER_EMAIL=$(gh api user --jq '.email // (.login + "@users.noreply.github.com")')
    git config --global user.name "${GITHUB_USER_NAME}"
    git config --global user.email "${GITHUB_USER_EMAIL}"
  # Extract username from git remote URL
  elif git remote get-url origin > /dev/null 2>&1; then
    REPO_OWNER=$(git remote get-url origin | sed -E 's#.*/([^/]+)/[^/]+\.git$#\1#')
    if [[ -n "${REPO_OWNER}" && "${REPO_OWNER}" != "$(git remote get-url origin)" ]]; then
      git config --global user.name "${REPO_OWNER}"
      git config --global user.email "${REPO_OWNER}@users.noreply.github.com"
    fi
  fi
fi
