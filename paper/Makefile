PANDOC = pandoc
SRC    = paper.md
BIB    = refs.bib
COMMON = $(SRC) --citeproc --bibliography $(BIB) --metadata link-citations=true

.PHONY: all pdf html clean

all: pdf

pdf:
	$(PANDOC) $(COMMON) --pdf-engine=xelatex -o paper.pdf

html:
	$(PANDOC) $(COMMON) --standalone --embed-resources -o paper.html

clean:
	rm -f paper.pdf paper.html
