#!/usr/bin/env bash
# Decide whether the pyproject version needs releasing.
#
# Writes `tag` and `release` to $GITHUB_OUTPUT and a line to $GITHUB_STEP_SUMMARY.
# On PRs (BASE_SHA set), fails if the PR changes the version to one that is
# already tagged: after merge, that bump would otherwise be silently skipped.
set -euo pipefail

version="$(uv version --short)"
tag="v${version}"
echo "tag=${tag}" >> "$GITHUB_OUTPUT"

if git ls-remote --exit-code --tags origin "refs/tags/${tag}" > /dev/null; then
  tagged=true
else
  tagged=false
fi

if [ -n "${BASE_SHA:-}" ]; then
  base_version="$(git show "${BASE_SHA}:pyproject.toml" \
    | python3 -c 'import sys, tomllib; print(tomllib.load(sys.stdin.buffer)["project"]["version"])')"
  if [ "${version}" != "${base_version}" ] && [ "${tagged}" = true ]; then
    echo "::error::Version changed ${base_version} -> ${version}, but ${tag} is already released"
    exit 1
  fi
fi

if [ "${tagged}" = true ]; then
  echo "release=false" >> "$GITHUB_OUTPUT"
  echo "${tag} is already released; nothing to publish." >> "$GITHUB_STEP_SUMMARY"
else
  echo "release=true" >> "$GITHUB_OUTPUT"
  if [ -n "${BASE_SHA:-}" ]; then
    echo "Merging this PR will publish **${tag}**." >> "$GITHUB_STEP_SUMMARY"
  fi
fi
