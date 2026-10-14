#!/usr/bin/env python3
"""仓库统一验证入口。

用法：python3 scripts/check_repo.py [docs|labs|third-party|all]
  docs        Markdown 相对链接有效、无模板占位符（四个可复用模板除外）
  labs        运行 labs/ 下的 unittest
  third-party 只读核对 third_party/ 与 repos.lock 一致
  all（默认）  docs + labs + third-party
退出码：0 通过；1 有检查失败；2 参数错误。
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 可复用模板：允许保留占位符，但必须带“模板状态”标记。
TEMPLATES = {
    "docs/superpowers/specs/YYYY-MM-DD-feature-design.md",
    "docs/superpowers/plans/YYYY-MM-DD-feature.md",
    "docs/references/reference.md",
    "docs/changelog/version.md",
}
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}|填写后删除")
FENCE_RE = re.compile(r"```.*?```", re.S)
# 行内代码中的外部源码引用，例如 `third_party/nccl/src/init.cc:120`。
TP_PATH_RE = re.compile(r"`(third_party/[^`\s]+)`")


def markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        # third_party 下只检查本仓库维护的 AGENTS.md，不检查外部源码。
        if rel.startswith("third_party/") and rel != "third_party/AGENTS.md":
            continue
        if rel.startswith(".git/"):
            continue
        yield path, rel


def check_docs():
    errors = []
    # 只有外部资料已拉取时才核对 `third_party/...` 路径，否则跳过并提示。
    third_party_ready = (ROOT / "third_party" / "nccl" / ".git").exists()
    if not third_party_ready:
        print("docs: third_party 未拉取，跳过源码路径检查（运行 scripts/fetch_third_party.sh）")
    for path, rel in markdown_files():
        text = path.read_text(encoding="utf-8")
        if rel in TEMPLATES:
            if "模板状态" not in text:
                errors.append(f"{rel}: 可复用模板缺少“模板状态”标记")
            continue
        prose = FENCE_RE.sub("", text)
        for m in PLACEHOLDER_RE.finditer(prose):
            errors.append(f"{rel}: 未处理的模板占位 {m.group(0)!r}")
        for m in TP_PATH_RE.finditer(prose):
            ref = m.group(1)
            # 跳过通配符/占位写法，以及 `third_party/<repo>@<commit>` 这种版本标注。
            if not third_party_ready or any(c in ref for c in "*<>{}$…@") or "..." in ref:
                continue
            ref = re.sub(r"(:\d+(-\d+)?|#L\d+.*)$", "", ref).rstrip("/.,;:")
            if not (ROOT / ref).exists():
                errors.append(f"{rel}: 源码路径不存在 {ref}")
        for target in LINK_RE.findall(prose):
            if re.match(r"^[a-z]+://|^mailto:|^#", target):
                continue
            dest = (path.parent / target.split("#", 1)[0]).resolve()
            if not dest.exists():
                errors.append(f"{rel}: 断链 {target}")
    for e in errors:
        print("FAIL", e)
    print(f"docs: {'failed' if errors else 'passed'} ({len(errors)} problems)")
    return not errors


def check_labs():
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "labs", "-t", "."]
    print("labs:", " ".join(cmd))
    ok = subprocess.run(cmd, cwd=ROOT).returncode == 0
    print(f"labs: {'passed' if ok else 'failed'}")
    return ok


def check_third_party():
    cmd = ["bash", "scripts/fetch_third_party.sh", "check"]
    ok = subprocess.run(cmd, cwd=ROOT).returncode == 0
    print(f"third-party: {'passed' if ok else 'failed'}")
    return ok


def main(argv):
    mode = argv[1] if len(argv) > 1 else "all"
    steps = {
        "docs": [check_docs],
        "labs": [check_labs],
        "third-party": [check_third_party],
        "all": [check_docs, check_labs, check_third_party],
    }
    if mode not in steps or len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    results = [step() for step in steps[mode]]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
