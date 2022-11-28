.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard *.ipynb)
CONVERTED := $(NOTEBOOKS:%.ipynb=docs/%.html)

TEMPLATE_OPTS := --TemplateExporter.extra_template_basedirs=nbconvert --template custom

docs/%.html : %.ipynb
> jupyter-nbconvert --to html ${TEMPLATE_OPTS} --output $@ $<

all: ${CONVERTED}

clean:
> rm -f ${CONVERTED}
