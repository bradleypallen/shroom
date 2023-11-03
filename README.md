# shroom

## Overview
 [SemEval-2024 Task-6 - Shared-task on Hallucinations and Related Observable Overgeneration Mistakes (SHROOM)](https://helsinki-nlp.github.io/shroom/)

## Approach

Given a datapoint from the validation data, we prompt GPT-4 using a zero-shot chain-of-thought to generate rationales arguing for or against the input ("hyp") of the datapoint as exhibiting hallucination, arguing from the perspective of five distinct personae, and then ask each persona to provide a final 'Hallucination' or 'Not Hallucination' answer. We then assign a final classification by majority vote across the personae, and provide a estimated probability for the datapoint exhibiting hallucination corresponding to the proportion of personae classifying the datapoint as exhibiting hallucination.

### Validation runs

| Classifier | Dataset | Run Notebook | Classification Results | Analysis Notebook |
| ---------- | ------- | ------------ | ---------------------- | ----------------- |
| [shroom_classifier_ensemble.py](https://github.com/bradleypallen/shroom/blob/e5ee7add48226c94ec1a53f30400c6a985ccb716/shroom_classifier_ensemble.py) | [train-v1.json](https://github.com/bradleypallen/shroom/blob/e5ee7add48226c94ec1a53f30400c6a985ccb716/trial-v1.json) | [shroom_experiment_trial-v1_ensemble.ipynb](https://github.com/bradleypallen/shroom/blob/e5ee7add48226c94ec1a53f30400c6a985ccb716/shroom_experiment_trial-v1_ensemble.ipynb) | [results_trial-v1_ensemble_version_4.json](https://github.com/bradleypallen/shroom/blob/e5ee7add48226c94ec1a53f30400c6a985ccb716/results_trial-v1_ensemble_version_5.json) | [shroom_trial-v1_metrics_ensemble.ipynb](https://github.com/bradleypallen/shroom/blob/e5ee7add48226c94ec1a53f30400c6a985ccb716/shroom_trial-v1_metrics_ensemble.ipynb) |


## Requirements
- Python 3.11 or higher

## Installation
``$ OPENAI_API_KEY = <your-OpenAI-API-key>``\
``$ pip install -r requirements.txt``

## License
MIT.
