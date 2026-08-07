#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_FILE="deploy.yml"
STATE_DIR=".github/deploy-state"
STATE_FILE="$STATE_DIR/deploy-state.json"
POLL_INTERVAL=3
POLL_ATTEMPTS=60

usage() {
  cat <<'USAGE'
Usage:
  .github/scripts/deploy.sh dev [ref]
  .github/scripts/deploy.sh prod [ref]

Defaults:
  dev  -> current branch, or develop when detached
  prod -> main

Set ALLOW_NON_MAIN_PROD=1 to deploy a non-main ref to prod.
USAGE
}

repo_name() {
  basename "$(git rev-parse --show-toplevel)"
}

project_urls() {
  case "$(repo_name)" in
    00-data-cleansing-with-sgd-main)
      DEV_URL="https://orbital-q.sakura.ne.jp/dev/Experiments--Data-Cleansing-with-SGD/"
      PROD_URL="https://orbital-q.sakura.ne.jp/Experiments--Data-Cleansing-with-SGD/"
      ;;
    *)
      echo "Unsupported project: $(repo_name)" >&2
      exit 1
      ;;
  esac
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

ensure_worktree_clean() {
  if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    echo "Worktree has uncommitted changes. Commit, stash, or remove them before deploy." >&2
    git status --short
    exit 1
  fi
}

default_ref() {
  if [ "$TARGET_ENV" = "prod" ]; then
    printf '%s\n' "main"
    return
  fi

  local branch
  branch="$(git branch --show-current 2>/dev/null || true)"
  if [ -n "$branch" ]; then
    printf '%s\n' "$branch"
  else
    printf '%s\n' "develop"
  fi
}

resolve_ref() {
  DEPLOY_REF="${REF:-$(default_ref)}"
  DEPLOY_COMMIT="$(git rev-parse "$DEPLOY_REF^{commit}")"
  DEPLOY_TREE="$(git rev-parse "$DEPLOY_REF^{tree}")"
}

ensure_prod_ref_is_main() {
  if [ "$TARGET_ENV" != "prod" ]; then
    return
  fi

  if [ "$DEPLOY_REF" = "main" ] || [ "${ALLOW_NON_MAIN_PROD:-}" = "1" ]; then
    return
  fi

  echo "Refusing prod deploy from non-main ref: $DEPLOY_REF" >&2
  echo "Set ALLOW_NON_MAIN_PROD=1 to override." >&2
  exit 1
}

state_tree() {
  local env="$1"
  jq -r --arg env "$env" '.[$env].tree // ""' "$STATE_FILE" 2>/dev/null || true
}

update_state() {
  local env="$1"
  local run_id="$2"
  local deployed_at
  local state_json
  local tmp

  deployed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  mkdir -p "$STATE_DIR"
  tmp="$STATE_FILE.tmp"

  if [ -f "$STATE_FILE" ] && jq empty "$STATE_FILE" >/dev/null 2>&1; then
    state_json="$(cat "$STATE_FILE")"
  else
    state_json="{}"
  fi

  printf '%s\n' "$state_json" | jq \
    --arg env "$env" \
    --arg tree "$DEPLOY_TREE" \
    --arg commit "$DEPLOY_COMMIT" \
    --arg ref "$DEPLOY_REF" \
    --arg run_id "$run_id" \
    --arg deployed_at "$deployed_at" \
    '.schema_version = 1 | .[$env] = {
      tree: $tree,
      commit: $commit,
      ref: $ref,
      workflow_run_id: $run_id,
      deployed_at: $deployed_at
    }' > "$tmp"
  mv "$tmp" "$STATE_FILE"
}

list_run_ids() {
  gh run list \
    --workflow "$WORKFLOW_FILE" \
    --event workflow_dispatch \
    --commit "$DEPLOY_COMMIT" \
    --limit 20 \
    --json databaseId \
    --jq '.[].databaseId'
}

known_run_id() {
  case " $1 " in
    *" $2 "*) return 0 ;;
    *) return 1 ;;
  esac
}

find_new_run_id() {
  local before_ids="$1"
  local run_id=""

  for _ in $(seq 1 "$POLL_ATTEMPTS"); do
    for id in $(list_run_ids); do
      if ! known_run_id "$before_ids" "$id"; then
        run_id="$id"
        break
      fi
    done

    if [ -n "$run_id" ]; then
      printf '%s\n' "$run_id"
      return
    fi

    sleep "$POLL_INTERVAL"
  done

  echo "Timed out waiting for workflow run to appear." >&2
  exit 1
}

smoke() {
  local env="$1"
  local url
  local expected
  local code

  if [ "$env" = "dev" ]; then
    url="$DEV_URL"
    expected="401"
  else
    url="$PROD_URL"
    expected="200"
  fi

  code="$(curl -sS -o /dev/null -w '%{http_code}' "$url")"
  if [ "$code" != "$expected" ]; then
    echo "Smoke check failed for $url: expected $expected, got $code" >&2
    exit 1
  fi
}

run_deploy() {
  local env="$1"
  local before_ids
  local run_id

  echo "Triggering $env deploy for $DEPLOY_REF ($DEPLOY_COMMIT)"
  before_ids="$(list_run_ids | tr '\n' ' ')"
  gh workflow run "$WORKFLOW_FILE" --ref "$DEPLOY_REF" -f environment="$env"
  run_id="$(find_new_run_id "$before_ids")"

  echo "Watching workflow run $run_id"
  gh run watch "$run_id" --exit-status --compact
  smoke "$env"
  update_state "$env" "$run_id"
  echo "Recorded $env deploy state for tree $DEPLOY_TREE"
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage >&2
  exit 1
fi

TARGET_ENV="$1"
REF="${2:-}"

case "$TARGET_ENV" in
  dev|prod) ;;
  *)
    usage >&2
    exit 1
    ;;
esac

require_command gh
require_command jq
require_command curl
ensure_worktree_clean
project_urls
resolve_ref
ensure_prod_ref_is_main

if [ "$TARGET_ENV" = "dev" ]; then
  run_deploy dev
  exit 0
fi

DEV_DEPLOYED_TREE="$(state_tree dev)"
if [ "$DEV_DEPLOYED_TREE" = "$DEPLOY_TREE" ] && [ -n "$DEV_DEPLOYED_TREE" ]; then
  echo "Dev already has tree $DEPLOY_TREE; skipping separate dev workflow."
else
  echo "Dev state mismatch or missing; deploying dev first."
  run_deploy dev
fi

run_deploy prod
