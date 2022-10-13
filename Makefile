.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard lecture_*.ipynb)
CONVERTED := $(NOTEBOOKS:%.ipynb=docs/%.html)

docs/%.html : %.ipynb
> jupyter-nbconvert --to html --output $@ $<

all: ${CONVERTED}

clean:
> rm -f ${CONVERTED}
