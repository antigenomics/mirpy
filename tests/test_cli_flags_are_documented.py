"""A command must not name a flag it does not have, and must not hide one it does.

The epilog of ``mir signature`` referred readers to ``--min-clonotypes`` for several releases while
the parser had no such option: the behaviour was real and reachable from Python, but the flag the
help text told you to use did not exist. ``--help`` cannot catch this -- it prints the epilog and
the option list side by side without comparing them.

Two directions, both checked here:

* every ``--flag`` mentioned in a subcommand's description or epilog is a real option of that
  subcommand;
* every real option of the signature and corpus subcommands appears in ``docs/cli.rst``, so the
  reference page cannot silently fall behind the parser.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pytest

from mir.cli import build_parser

CLI_RST = pathlib.Path(__file__).resolve().parents[1] / "docs" / "cli.rst"
FLAG = re.compile(r"--[a-z][a-z0-9-]+")

# Flags named in prose that belong to the *other* half's command, not to this parser.
FOREIGN = {"--corpus-dir"}
INVOCATION = re.compile(r"^\s*(?:mir|vdjtools)\s+([a-z-]+)")


def _own_prose(name: str, parser: argparse.ArgumentParser) -> str:
    """Description and epilog, minus example lines that invoke a *different* command.

    An epilog may legitimately show a sibling command -- ``mir signature``'s shows how to build a
    corpus with ``mir corpus --smoke`` -- and those flags belong to that sibling, not here.
    """
    leaf = name.split()[-1]
    keep = []
    for line in " \n".join(filter(None, (parser.description, parser.epilog))).splitlines():
        m = INVOCATION.match(line)
        if m and m.group(1) != leaf:
            continue
        keep.append(line)
    return "\n".join(keep)


def _subparsers(parser: argparse.ArgumentParser):
    """Every (dotted name, parser) pair, walking nested subparsers."""
    out = [("mir", parser)]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                for child_name, child in _subparsers(sub):
                    suffix = "" if child_name == "mir" else f" {child_name.removeprefix('mir ')}"
                    out.append((f"mir {name}{suffix}".rstrip(), child))
    return out


def _options(parser: argparse.ArgumentParser) -> set[str]:
    return {s for a in parser._actions for s in a.option_strings if s.startswith("--")}


ALL = _subparsers(build_parser())
LEAVES = [(name, p) for name, p in ALL if not any(
    isinstance(a, argparse._SubParsersAction) for a in p._actions)]

assert LEAVES, "no leaf subcommands found"


@pytest.mark.parametrize("name,parser", LEAVES, ids=[n for n, _ in LEAVES])
def test_every_flag_the_help_text_names_actually_exists(name, parser) -> None:
    named = set(FLAG.findall(_own_prose(name, parser))) - FOREIGN
    missing = sorted(named - _options(parser))
    assert not missing, (
        f"{name}: help text refers to {missing}, which the parser does not define. "
        f"Either add the option or stop naming it."
    )


@pytest.mark.parametrize("name", ["mir signature", "mir corpus"])
def test_the_reference_page_documents_every_option(name) -> None:
    parser = dict(LEAVES)[name]
    text = CLI_RST.read_text()
    undocumented = sorted(o for o in _options(parser) if o not in text and o != "--help")
    assert not undocumented, (
        f"{name}: options absent from docs/cli.rst: {undocumented}. "
        f"A reference page that lags the parser is worse than no reference page."
    )


def test_the_check_can_actually_fail() -> None:
    """A guard: a parser whose epilog names a flag it lacks must be caught."""
    p = argparse.ArgumentParser(epilog="pass --nonesuch to enable it")
    p.add_argument("--real")
    named = set(FLAG.findall(p.epilog)) - FOREIGN
    assert sorted(named - _options(p)) == ["--nonesuch"]
