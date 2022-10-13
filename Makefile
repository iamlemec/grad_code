.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard lecture_*.ipynb)
CONVERTED := $(NOTEBOOKS:%.ipynb=html/%.html)

html/%.html : %.ipynb
> jupyter-nbconvert --to html --output $@ $<

all: ${CONVERTED}

clean:
> rm -f ${CONVERTED}
