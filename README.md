# shroom

## Overview
 [SemEval-2024 Task-6 - Shared-task on Hallucinations and Related Observable Overgeneration Mistakes (SHROOM)](https://helsinki-nlp.github.io/shroom/)

## Approach

Given a datapoint from the validation data, we prompt a large language model to determine whether the input ("hyp") of the datapoint as exhibiting hallucination, arguing from the perspective of a task-specific persona. We perform temperature sampling and then assign a final classification by majority vote across multiple invocations of the model, and provide a estimated probability for the datapoint exhibiting hallucination.

### Validation runs (in ```prod``` directory)

| version                                | date       |   agnostic_acc |   agnostic_rho |   aware_acc |   aware_rho |   avg_acc |   avg_rho |
|:---------------------------------------|:-----------|---------------:|---------------:|------------:|------------:|----------:|----------:|
| mistral-baseline                       | 2023-12-23 |       0.649299 |       0.380141 |    0.706587 |    0.460958 |  0.677943 |  0.420549 |
| gpt-4-baseline_v1                      | 2024-01-09 |       0.801603 |       0.679521 |    0.762475 |    0.555941 |  0.782039 |  0.617731 |
| gpt-3.5-turbo-baseline_v1              | 2024-01-09 |       0.725451 |       0.54904  |    0.712575 |    0.541777 |  0.719013 |  0.545408 |
| gpt-3.5-turbo-persona-examples         | 2024-01-10 |       0.671343 |       0.493688 |    0.712575 |    0.524608 |  0.691959 |  0.509148 |
| gpt-3.5-turbo-persona                  | 2024-01-10 |       0.715431 |       0.575568 |    0.720559 |    0.535034 |  0.717995 |  0.555301 |
| gpt-3.5-turbo-baseline_v3_temp-1.2     | 2024-01-11 |       0.735471 |       0.600531 |    0.722555 |    0.516681 |  0.729013 |  0.558606 |
| gpt-3.5-turbo-temp-0.2                 | 2024-01-11 |       0.711423 |       0.488756 |    0.742515 |    0.528502 |  0.726969 |  0.508629 |
| gpt-3.5-turbo-baseline_v3              | 2024-01-11 |       0.745491 |       0.584922 |    0.718563 |    0.508341 |  0.732027 |  0.546631 |
| gpt-3.5-turbo-temp-1.2                 | 2024-01-11 |       0.709419 |       0.519809 |    0.752495 |    0.582654 |  0.730957 |  0.551231 |
| gpt-3.5-turbo-conservative             | 2024-01-11 |       0.701403 |       0.540491 |    0.734531 |    0.568859 |  0.717967 |  0.554675 |
| gpt-3.5-turbo-baseline-v2              | 2024-01-11 |       0.701403 |       0.52644  |    0.742515 |    0.569994 |  0.721959 |  0.548217 |
| gpt-4-baseline_v3_temp-1.2             | 2024-01-12 |       **0.821643** |       **0.722481** |    **0.782435** |    **0.627895** |  **0.802039** |  **0.675188** |
| gpt-3.5-turbo-baseline_v4_persona      | 2024-01-13 |       0.753507 |       0.617318 |    0.736527 |    0.570115 |  0.745017 |  0.593717 |
| gpt-3.5-turbo-baseline_v3_persona      | 2024-01-13 |       0.757515 |       0.612149 |    0.736527 |    0.592091 |  0.747021 |  0.60212  |
| gpt-3.5-turbo-baseline_v4              | 2024-01-13 |       0.743487 |       0.606295 |    0.748503 |    0.597773 |  0.745995 |  0.602034 |
| gpt-3.5-turbo-baseline_v3_temp-1.5     | 2024-01-13 |       0.717435 |       0.541716 |    0.708583 |    0.534737 |  0.713009 |  0.538226 |
| gpt-4-1106-preview-baseline_v4_persona | 2024-01-13 |       0.815631 |       0.709876 |    0.768463 |    0.612371 |  0.792047 |  0.661123 |
| gpt-3.5-turbo-baseline_v4_persona      | 2024-01-14 |       0.739479 |       0.585354 |    0.746507 |    0.615651 |  0.742993 |  0.600503 |
| gpt-4-1106-preview-baseline_v4_persona | 2024-01-14 |       0.815631 |       0.7101   |    0.766467 |    0.620185 |  0.791049 |  0.665143 |

## Requirements
- Python 3.11 or higher

## Installation
``$ OPENAI_API_KEY = <your-OpenAI-API-key>``\
``$ pip install -r requirements.txt``

## License
MIT.
