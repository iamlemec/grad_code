.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard *.ipynb)
CONVERTED := $(NOTEBOOKS:%.ipynb=docs/%.html)

TEMPLATE_OPTS := --TemplateExporter.extra_template_basedirs=nbconvert --template custom
METADATA_OPTS := --ClearMetadataPreprocessor.enabled=True

docs/%.html : %.ipynb
> jupyter-nbconvert ${METADATA_OPTS} --to html ${TEMPLATE_OPTS} --output $@ $<

all: ${CONVERTED}

clean:
> rm -f ${CONVERTED}
