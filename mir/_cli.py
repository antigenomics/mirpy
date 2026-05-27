"""mirpy command-line interface.

Entry point for ``mirpy <command> <subcommand>`` invocations installed by pip.

Commands
--------
mirpy install skills [--target DIR] [--force]
    Copy bundled agent skills into the target project directory under both
    ``skills/mirpy/`` (Claude Code) and ``.agents/skills/mirpy/``
    (VS Code / GitHub Copilot / OpenAI Codex / Gemini CLI and any agent
    that follows the agentskills.io convention).

mirpy cache status
    Show the contents and disk usage of the mirpy controls cache.

mirpy cache purge [--yes]
    Delete all cached control files.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

_CONTROL_ENV = "MIRPY_CONTROL_DIR"
_SKILLS_DIR = Path(__file__).parent / "_skills"


def _get_cache_dir() -> Path:
    env = os.environ.get(_CONTROL_ENV)
    return Path(env) if env else Path.home() / ".cache" / "mirpy" / "controls"


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


# Agent → callable that derives the skills subfolder from the project root
_SKILL_DESTINATIONS: dict[str, object] = {
    "Claude Code": lambda root: root / "skills",
    "VS Code / GitHub Copilot / OpenAI Codex / Gemini CLI": lambda root: root / ".agents" / "skills",
}

# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _install_skills(target_dir: str, force: bool) -> None:
    target = Path(target_dir).resolve()
    source = _SKILLS_DIR / "mirpy"
    if not source.exists():
        print(f"error: bundled skills not found at {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Installing mirpy agent skills into {target}\n")
    installed = 0
    for label, dest_fn in _SKILL_DESTINATIONS.items():
        dest: Path = dest_fn(target) / "mirpy"
        if dest.exists():
            if not force:
                print(f"  — {label}: already present at {dest}  (use --force to overwrite)")
                continue
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, symlinks=False)
        print(f"  ✓ {label}: {dest}")
        installed += 1

    if installed:
        print(
            "\nSkills ready for Claude Code, VS Code, GitHub Copilot, OpenAI Codex,\n"
            "Gemini CLI, and any agent following the agentskills.io convention."
        )
    else:
        print("\nNo skills were installed (all destinations already exist).")


def _cache_status() -> None:
    cache_dir = _get_cache_dir()
    print(f"Controls cache: {cache_dir}")
    if not cache_dir.exists():
        print("  (empty — no cached files)")
        return

    files = sorted(p for p in cache_dir.rglob("*") if p.is_file())
    if not files:
        print("  (empty — no cached files)")
        return

    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        print(f"  {path.relative_to(cache_dir)}  ({_fmt_size(size)})")

    print(f"\n  Total: {_fmt_size(total)} in {len(files)} file(s)")


def _cache_purge(yes: bool) -> None:
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    if not files:
        print("Cache is empty. Nothing to purge.")
        return

    total = sum(p.stat().st_size for p in files)
    print(f"Will delete {len(files)} file(s) ({_fmt_size(total)}) from {cache_dir}")

    if not yes:
        try:
            answer = input("Confirm purge? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer != "y":
            print("Aborted.")
            return

    shutil.rmtree(cache_dir)
    print(f"Deleted {cache_dir}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mirpy",
        description=(
            "mirpy — Immunosequencing Algorithms Laboratory (ISALGO lab)\n"
            "AIRR-seq toolkit for TCR/BCR repertoire analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True

    # ------------------------------------------------------------------ #
    # install
    # ------------------------------------------------------------------ #
    p_install = subs.add_parser("install", help="Install mirpy resources")
    inst_subs = p_install.add_subparsers(dest="resource", metavar="RESOURCE")
    inst_subs.required = True

    p_skills = inst_subs.add_parser(
        "skills",
        help="Install mirpy agent skills for Claude Code, Copilot, Codex, Gemini CLI, …",
    )
    p_skills.add_argument(
        "--target", "-t",
        default=".",
        metavar="DIR",
        help="Target project directory (default: current directory)",
    )
    p_skills.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing skill installations",
    )

    # ------------------------------------------------------------------ #
    # cache
    # ------------------------------------------------------------------ #
    p_cache = subs.add_parser("cache", help="Manage the mirpy file cache")
    cache_subs = p_cache.add_subparsers(dest="action", metavar="ACTION")
    cache_subs.required = True

    cache_subs.add_parser("status", help="Show cache contents and disk usage")

    p_purge = cache_subs.add_parser("purge", help="Delete all cached control files")
    p_purge.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt",
    )

    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "install":
        if args.resource == "skills":
            _install_skills(args.target, args.force)

    elif args.command == "cache":
        if args.action == "status":
            _cache_status()
        elif args.action == "purge":
            _cache_purge(args.yes)


if __name__ == "__main__":
    main()
