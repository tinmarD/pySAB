# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = SABAnalysis
SOURCEDIR     = ..
BUILDDIR      = _build
RSTDIR 	      = rst

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
SPHINXOPTS = -d _build/doctrees $(PAPEROPT_$(PAPER)) .


.PHONY: help clean html Makefile


help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  clean     to clean the build directory"

clean:
	-rm -rf $(BUILDDIR)
	-rm -rf auto_examples/
	-rm -rf $(RSTDIR)

html:
	sphinx-apidoc -o $(RSTDIR) $(SOURCEDIR)
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(BUILDDIR)/html


