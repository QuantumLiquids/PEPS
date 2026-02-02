#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
USER_DOCS = ROOT / "docs" / "user"


FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class LinkIssue:
  file: Path
  target: str
  reason: str


def strip_code(markdown: str) -> str:
  markdown = FENCE_RE.sub("", markdown)
  markdown = INLINE_CODE_RE.sub("", markdown)
  return markdown


def iter_markdown_files(root: Path) -> list[Path]:
  return sorted(p for p in root.rglob("*.md") if p.is_file())


def rel_set(root: Path, lang: str) -> set[str]:
  base = root / lang
  return {str(p.relative_to(base)) for p in iter_markdown_files(base)}


def check_parity(user_docs: Path) -> list[str]:
  en = rel_set(user_docs, "en")
  zh = rel_set(user_docs, "zh")
  only_en = sorted(en - zh)
  only_zh = sorted(zh - en)
  errs: list[str] = []
  if only_en:
    errs.append("Paths only in docs/user/en:")
    errs.extend([f"  {p}" for p in only_en])
  if only_zh:
    errs.append("Paths only in docs/user/zh:")
    errs.extend([f"  {p}" for p in only_zh])
  return errs


def is_external(target: str) -> bool:
  lower = target.strip().lower()
  return (
    lower.startswith("http://")
    or lower.startswith("https://")
    or lower.startswith("mailto:")
    or lower.startswith("#")
  )


def normalize_target(target: str) -> str:
  target = target.strip()
  # Drop URL fragment for filesystem existence checks.
  if "#" in target:
    target = target.split("#", 1)[0]
  # Drop query string just in case.
  if "?" in target:
    target = target.split("?", 1)[0]
  return target.strip()


def check_links(user_docs: Path) -> list[LinkIssue]:
  issues: list[LinkIssue] = []
  for md in iter_markdown_files(user_docs):
    text = md.read_text(encoding="utf-8")
    text = strip_code(text)
    for m in MD_LINK_RE.finditer(text):
      raw = m.group(1).strip()
      if not raw or is_external(raw):
        continue
      target = normalize_target(raw)
      if not target:
        continue
      if target.startswith("/"):
        # Repo-absolute is not supported by GitHub in the same way; treat as project-root-relative.
        resolved = ROOT / target.lstrip("/")
      else:
        resolved = (md.parent / target).resolve()
      if not resolved.exists():
        issues.append(LinkIssue(md, raw, f"missing path: {resolved}"))
  return issues


def main() -> int:
  parser = argparse.ArgumentParser(description="Validate docs/user/ EN-ZH parity and local links.")
  parser.add_argument("--check-links", action="store_true", help="Also validate relative markdown links.")
  args = parser.parse_args()

  if not USER_DOCS.exists():
    print(f"ERROR: docs/user directory not found at: {USER_DOCS}", file=sys.stderr)
    return 2

  parity_errs = check_parity(USER_DOCS)
  if parity_errs:
    print("\n".join(parity_errs), file=sys.stderr)
    return 1

  if args.check_links:
    issues = check_links(USER_DOCS)
    if issues:
      print("Broken relative links found:", file=sys.stderr)
      for it in issues:
        rel = it.file.relative_to(ROOT)
        print(f"- {rel}: ({it.target}) -> {it.reason}", file=sys.stderr)
      return 1

  print("OK")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

