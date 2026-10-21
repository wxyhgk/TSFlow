#!/usr/bin/env python3
"""Export the current source tree without local research or Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Keep this explicit: a new case folder must not enter a public snapshot merely
# because someone forgot to add a .gitignore rule.
SOURCE_PATHS = (
    ".gitignore", "README.md", "README.en.md", "AI.md", "PROJECT_STATUS.md", "LICENSE", "NOTICE",
    "backend/app", "backend/tests", "backend/pyproject.toml",
    "backend/README.md", "backend/.env.example",
    # The frontend remains a local workbench until its build and browser flow
    # are independently verified for publication.
    "packages/uma-workflows", "scripts", "tools", "examples", "assets",
    "docs/README.md", "docs/quickstart.md", "docs/architecture.md",
    "docs/project-layout.md", "docs/transition-state-workflow.md",
    "docs/github-release.md", "docs/calculator-providers",
    "model-assets/README.md", "model-assets/UMA/LICENSE",
    "model-assets/UMA/USE_POLICY.md", "model-assets/UMA/references",
)
MAX_FILE_BYTES = 5 * 1024 * 1024


def git(*args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], input=input_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def source_files() -> list[Path]:
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS)
    if listed.returncode:
        raise RuntimeError(listed.stderr.decode())
    names = sorted(set(listed.stdout.split(b"\0")) - {b""})
    ignored = git("check-ignore", "--no-index", "--stdin", "-z", input_bytes=b"\0".join(names) + b"\0")
    if ignored.returncode not in (0, 1):
        raise RuntimeError(ignored.stderr.decode())
    exclusions = set(ignored.stdout.split(b"\0"))
    files = []
    for name in names:
        if name in exclusions:
            continue
        path = ROOT / name.decode()
        if path.is_symlink():
            raise RuntimeError(f"Review symlink before export: {path.relative_to(ROOT)}")
        if not path.is_file():  # Deleted tracked files are not part of the snapshot.
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError(f"Review file larger than 5 MiB: {path.relative_to(ROOT)}")
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New destination directory; existing directories are refused")
    parser.add_argument("--list", action="store_true", help="List selected files without creating a snapshot")
    args = parser.parse_args()
    if not args.list and args.output is None:
        parser.error("provide --list or --output DIRECTORY")
    files = source_files()
    if args.list:
        for path in files:
            print(path.relative_to(ROOT))
        print(f"{len(files)} files; {sum(p.stat().st_size for p in files):,} bytes")
        return
    destination = args.output.expanduser().resolve()
    # Never overwrite an existing directory or export inside a source subtree.
    if destination == ROOT or any(destination.is_relative_to(ROOT / p) for p in SOURCE_PATHS):
        parser.error("destination must be outside the selected source paths (use outputs/)")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({
            "path": relative.as_posix(), "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    head = git("rev-parse", "HEAD")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": head.stdout.decode().strip() if head.returncode == 0 else None,
        "source_has_local_changes": bool(git("status", "--porcelain").stdout),
        "scope": "Current working-tree source snapshot; no Git history, weights or local research",
        "file_count": len(manifest), "files": manifest,
    }
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(files)} files to {destination}")
    print("No Git repository, commit or remote was created in the export.")


if __name__ == "__main__":
    main()
