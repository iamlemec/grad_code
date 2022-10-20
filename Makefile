.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard lecture_*.ipynb)
CONVERTED := $(NOTEBOOKS:%.ipynb=templates/%.html)
RENDERED := $(NOTEBOOKS:%.ipynb=docs/%.html)

templates/%.html : %.ipynb
> jupyter-nbconvert --to html --template basic --output $@ $<

docs/%.html : templates/%.html
> python3 render.py $(<F) $@

all: ${CONVERTED} ${RENDERED}

clean:
> rm -f ${CONVERTED} ${RENDERED}
