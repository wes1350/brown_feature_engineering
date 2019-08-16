from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.container.dataset import Dataset
from d3m.runtime import Runtime
import os

# Loading problem description.
problem_description = problem.parse_problem_description('problemDoc.json')

# Loading dataset.
path = 'file://{uri}'.format(uri=os.path.abspath('datasetDoc.json'))
dataset = Dataset.load(dataset_uri=path)

# Loading pipeline description file.
with open('pipeline_description.json', 'r') as file:
    pipeline_description = pipeline_module.Pipeline.from_json(string_or_file=file)

# Creating an instance on runtime with pipeline description and problem description.
runtime = Runtime(pipeline=pipeline_description, problem_description=problem_description, context=metadata_base.Context.TESTING)

# Fitting pipeline on input dataset.
fit_results = runtime.fit(inputs=[dataset])
fit_results.check_success()

# Producing results using the fitted pipeline.
produce_results = runtime.produce(inputs=[dataset])
produce_results.check_success()

print(produce_results.values)