.RECIPEPREFIX = >

NOTEBOOKS := $(wildcard notebooks/*.ipynb)
CONVERTED := $(NOTEBOOKS:notebooks/%.ipynb=docs/%.html)

TEMPLATE_OPTS := --TemplateExporter.extra_template_basedirs=nbconvert --template custom
METADATA_OPTS := --ClearMetadataPreprocessor.enabled=True

docs/%.html : notebooks/%.ipynb
> jupyter-nbconvert ${METADATA_OPTS} --to html ${TEMPLATE_OPTS} --output-dir . --output $@ $<

all: ${CONVERTED}

clean:
> rm -f ${CONVERTED}
