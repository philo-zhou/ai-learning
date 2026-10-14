#!/usr/bin/env bash
# 按 third_party/repos.lock 克隆或核对外部学习资料。
# 用法：scripts/fetch_third_party.sh [fetch|check]
#   fetch（默认）：缺失则浅克隆并检出锁定 commit；已存在则只核对，不覆盖本地内容。
#   check：只读核对每个目录是否存在且 HEAD 等于锁定 commit。
# 退出码：0 全部一致；1 有缺失/不一致/克隆失败；2 参数错误。
set -euo pipefail

mode="${1:-fetch}"
case "$mode" in
  fetch|check) ;;
  *) echo "unknown mode: $mode (expected fetch|check)" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"
lock="$root/third_party/repos.lock"
status=0

while read -r name url commit; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue
  dir="$root/third_party/$name"
  if [[ ! -d "$dir/.git" ]]; then
    if [[ "$mode" == check ]]; then
      echo "MISSING  $name"; status=1; continue
    fi
    echo "CLONE    $name"
    if ! { git init -q "$dir" \
        && git -C "$dir" remote add origin "$url" \
        && git -C "$dir" fetch -q --depth 1 origin "$commit" \
        && git -C "$dir" checkout -q FETCH_HEAD; }; then
      echo "FAILED   $name" >&2; status=1; continue
    fi
  fi
  head="$(git -C "$dir" rev-parse HEAD)"
  if [[ "$head" == "$commit" ]]; then
    echo "OK       $name ${commit:0:8}"
  else
    echo "MISMATCH $name HEAD=${head:0:8} lock=${commit:0:8}"; status=1
  fi
done < "$lock"

exit "$status"
