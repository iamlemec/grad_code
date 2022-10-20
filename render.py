# jupyter notebook renderer

import sys
from jinja2 import Environment, FileSystemLoader

# get command line arguments
_, input_file, output_file = sys.argv

# point to templates directory
load = FileSystemLoader('templates')
env = Environment(loader=load)

# read in template
template = env.get_template('template.html')

# render the template
rendered = template.render(notebook=input_file)

# write the result to disk
with open(output_file, 'w') as ofile:
    ofile.write(rendered)
