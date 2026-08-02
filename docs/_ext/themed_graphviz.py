"""Theme-aware graphviz: one dot source, rendered once per colour scheme.

``sphinx.ext.graphviz`` bakes colours into the SVG and embeds it with ``<object>``, so page CSS
cannot reach inside to recolour it and a single diagram must pick one palette. This directive
renders the same source twice — once per palette below — wrapping each in a container that
``custom.css`` shows or hides according to the active theme (pydata-sphinx-theme sets
``data-theme`` on ``<html>``).

Write ``$token`` placeholders in the dot source; :data:`PALETTES` defines them. Literal ``$`` is
``$$``. An unknown token is a build error rather than a silently unstyled diagram.

Usage::

    .. themed-graphviz::
       :alt: what the diagram shows

       digraph d {
         bgcolor="transparent";
         node [color="$line", fontcolor="$ink", fillcolor="$fill_key"];
       }
"""

from __future__ import annotations

from string import Template

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.ext.graphviz import graphviz
from sphinx.util.docutils import SphinxDirective

#: Every diagram is drawn on a transparent background, so only ink and fills change.
PALETTES = {
    "light": {
        "ink": "#22303c",        # box text
        "ink_soft": "#5b6b7a",   # edge labels, captions, note text
        "line": "#8899a6",       # borders and edges
        "fill": "#f4f6f8",       # neutral box
        "fill_key": "#e8f0fe",   # the object being constructed
        "fill_alt": "#fdf0e3",   # an estimated / uncertain quantity
        "fill_ok": "#e6f4ea",    # a result
        "fill_note": "#ffffff",  # the white annotation card
        "good": "#2e7d4f",       # preserved
        "bad": "#b3452c",        # destroyed
    },
    "dark": {
        "ink": "#d7dee6",
        "ink_soft": "#9aa8b5",
        "line": "#5c6b7a",
        "fill": "#1f262e",
        "fill_key": "#1d2a3a",
        "fill_alt": "#33291c",
        "fill_ok": "#18301f",
        "fill_note": "#151a20",
        "good": "#6fbf8b",
        "bad": "#e08b72",
    },
}


class ThemedGraphviz(SphinxDirective):
    """``graphviz`` with ``$token`` palette substitution, emitted once per palette."""

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "alt": directives.unchanged,
        "align": lambda a: directives.choice(a, ("left", "center", "right")),
        "layout": directives.unchanged,
        "class": directives.class_option,
    }

    def run(self) -> list[nodes.Node]:
        source = "\n".join(self.content)
        if not source.strip():
            return [self.state_machine.reporter.warning(
                "Ignoring themed-graphviz directive without content.", line=self.lineno)]
        out: list[nodes.Node] = []
        for theme, palette in PALETTES.items():
            try:
                code = Template(source).substitute(palette)
            except KeyError as exc:
                raise self.error(
                    f"themed-graphviz: unknown palette token {exc}; "
                    f"known tokens: {', '.join(sorted(palette))}"
                ) from None
            node = graphviz()
            node["code"] = code
            node["options"] = {"docname": self.env.docname}
            if "layout" in self.options:
                node["options"]["graphviz_dot"] = self.options["layout"]
            if "alt" in self.options:
                node["alt"] = self.options["alt"]
            node["align"] = self.options.get("align", "center")
            node["classes"] = self.options.get("class", [])
            out.append(nodes.container("", node, classes=["mir-diagram", f"mir-only-{theme}"]))
        return out


def setup(app):
    app.setup_extension("sphinx.ext.graphviz")
    app.add_directive("themed-graphviz", ThemedGraphviz)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
