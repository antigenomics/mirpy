# Minimal Sphinx Makefile for local docs builds.
SPHINXOPTS ?=
SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = _build
NB_SRC = ../notebooks
NB_DST = notebooks

.PHONY: help html clean sync-notebooks

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Ensure every notebook in notebooks/ has a matching symlink in docs/notebooks/.
# Safe to run repeatedly: skips existing correct symlinks, only creates/fixes missing ones.
sync-notebooks:
	@python3 sync_notebooks.py

html: sync-notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"
