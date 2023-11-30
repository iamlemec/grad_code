.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard code/*.ipynb)
CONVERTED := $(NOTEBOOKS:code/%.ipynb=docs/%.html)

TEMPLATE_OPTS := --TemplateExporter.extra_template_basedirs=nbconvert --template custom
METADATA_OPTS := --ClearMetadataPreprocessor.enabled=True

docs/%.html : code/%.ipynb
> jupyter-nbconvert ${METADATA_OPTS} --to html ${TEMPLATE_OPTS} --output-dir . --output $@ $<

all: ${CONVERTED}

clean:
> rm -f ${CONVERTED}
