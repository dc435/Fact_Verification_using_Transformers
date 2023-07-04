# Fact Verification and Extraction of Climate-Related Claims

## Overview

This is a report and codebase prepared by the author as part of a University natural language processing challenge.

The challenge involved building an NLP pipeline to verify climate-related claims and extract supporting evidence. The dataset was amended from ClimateFever (https://huggingface.co/datasets/climate_fever).

I built a three step pipeline:

1. Evidence shortlisting using a 'distilbert' model and spaCy named-entities;
2. Evidence ranking point-wise comparisons using 'roberta-large' model;
3. Claim classification natural language inference using 'roberta-large-mnli' model.

Built in python using huggingface transformers, spaCy, pandas and others.

My results and discussion are set out in 'report.pdf'.

## Structure

The code base is structured as follows:

1. *main.ipynb* is the main notebook for inferences. It imports functions from the helper .py files. It contains an args object containing the hyperparameters.
2. The *.py* files (A,B,C) contain the helper methods.
3. Other *.ipynb* files contain the code for training / finetuning the models.
4. The subfolders have been retained, but their content deleted (as instructed). They contain the original data (/data), trained model weights (/models), final inference output (/output) and pickle files used during developemt (/pickles).

The code needs to be run sequentially as shown in main.ipynb. In particular, the models in steps B and C need to be separately trained and the weights saved before they can be accessed for inference in those steps.