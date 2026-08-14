#!/usr/bin/env python3
"""Classify the Higgs-Audio native-codegen diff by review boundary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_QWEN3_DIAGNOSTIC = {
    "pegainfer-qwen3/src/batch_decode.rs",
    "pegainfer-qwen3/src/executor.rs",
    "pegainfer-qwen3/src/lib.rs",
}

EXPECTED_SERVER_WIRING = {
    "Cargo.lock",
    "pegainfer-server/Cargo.toml",
    "pegainfer-server/src/main.rs",
}

EXPECTED_SHARED_NONE_PREFIXES = (
    "pegainfer-core/",
    "pegainfer-kernels/",
    "pegainfer-frontend/",
    "pegainfer-kv-cache/",
)

ALLOWED_THIRD_PARTY_DIRTY = {
    "pegainfer-kernels/third_party/DeepGEMM",
    "pegainfer-kernels/third_party/FlashMLA",
    "pegainfer-kernels/third_party/flashinfer",
}


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    )


def diff_paths(repo: Path, ref: str) -> list[str]:
    paths: set[str] = set()
    for line in run_git(repo, "diff", "--name-only", ref).splitlines():
        if line.strip():
            paths.add(line.strip())
    for line in run_git(repo, "ls-files", "--others", "--exclude-standard").splitlines():
        if line.strip():
            paths.add(line.strip())
    return sorted(paths)


def classify(path: str) -> str:
    if path.startswith("pegainfer-higgs-audio/"):
        return "higgs-local"
    if path.startswith("docs/models/higgs-audio/"):
        return "doc-tool"
    if path == "docs/index.md":
        return "doc-tool"
    if path.startswith("tools/higgs/"):
        return "doc-tool"
    if path in EXPECTED_QWEN3_DIAGNOSTIC:
        return "qwen3-diagnostic"
    if path in EXPECTED_SERVER_WIRING:
        return "server-feature-wiring"
    if path in ALLOWED_THIRD_PARTY_DIRTY:
        return "unexpected-third-party-dirty"
    if path.startswith(EXPECTED_SHARED_NONE_PREFIXES):
        return "unexpected-shared"
    return "unexpected"


def render_markdown(groups: dict[str, list[str]]) -> str:
    lines = [
        "# Higgs-Audio native-codegen diff classification",
        "",
        "| Classification | Files |",
        "| --- | --- |",
    ]
    for name in sorted(groups):
        files = "<br>".join(f"`{path}`" for path in groups[name])
        lines.append(f"| `{name}` | {files} |")
    lines.append("")
    lines.append("Review notes:")
    lines.append("- `higgs-local` and `doc-tool` are the expected owners for this slice.")
    lines.append(
        "- `qwen3-diagnostic` is expected only for the narrow embedding-fed retained-hidden API."
    )
    lines.append(
        "- `server-feature-wiring` is expected only for feature-gated registration and config hints."
    )
    lines.append(
        "- `unexpected-third-party-dirty`, `unexpected-shared`, and `unexpected` require cleanup or explicit maintainer-facing rationale before PR."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--fail-on-unexpected",
        action="store_true",
        help="Exit non-zero if unexpected/shared/third-party dirty files are present.",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    paths = diff_paths(repo, args.ref)
    if not paths:
        print("higgs native codegen diff classification: empty")
        return 0

    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[classify(path)].append(path)

    text = render_markdown(groups)
    if args.markdown_out:
        args.markdown_out.write_text(text + "\n")
    print(text)

    bad = [
        path
        for name in ("unexpected-third-party-dirty", "unexpected-shared", "unexpected")
        for path in groups.get(name, [])
    ]
    if bad and args.fail_on_unexpected:
        print("unexpected files present:", file=sys.stderr)
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
