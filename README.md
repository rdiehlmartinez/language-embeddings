# Extraction and Analysis of Language Embeddings 

Tools for extracting and analyzing language embeddings. 

Language embeddings are extracted by running a [task2vec](https://arxiv.org/abs/1902.03545) 
algorithm on a pre-trained XLM-r model. 

Concretely, for a given dataset in a language we pass the data through the pre-trained 
language model and compute the Fisher Information Matrix (FIM) for each layer of the model 
per each data point. We then globally pool the FIM for each language dataset - as in the 
task2vec paper we take the diagonal of the FIM as the embedding vector. 

We also provide tooling for analyzing the resulting embeddings visually (PCA plots). 

## Loading in data 
Run 
    ./data/get_data.sh && python3 ./data/preprocess.py 

This will load in the dataset and generate a preprocessed dataset in the /data folder. 

## Running embedding extraction 
Run 
    vitualenv -p python3 env && source env/bin/activate
    pip install -r requirements/env_embeddings_extractor_requirements.txt 
    python3 embeddings_extractor.py 

This will dump out a pickle file in the root directory with the computed embeddings. 
The pickle file is a dictionary object of the form: 
{language -> dictionary of embeddings {layer: embedding}}.

## Plotting Anaylsis 
Analysis is done in a jupyter notebook that can be found in the analysis folder. 
In
    requirements/env_plot_requirements.txt

we provide the requirements for the environment required to run analysis with our 
jupyter notebook.

