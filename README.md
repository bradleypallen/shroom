# shroom

## Overview
 [SemEval-2024 Task-6 - Shared-task on Hallucinations and Related Observable Overgeneration Mistakes (SHROOM)](https://helsinki-nlp.github.io/shroom/)

## Approach

Given a datapoint from the validation data, we prompt a large language model to determine whether the output ("hyp") of the datapoint exhibits hallucination, arguing from the perspective of a task-specific persona. We perform temperature sampling and then assign a final classification by majority vote across multiple invocations of the model to compute a estimated probability that the datapoint exhibits hallucination.

## Versions

| Name | Description | Source |
| ---- | ----------- | ----------------- |
| baseline | The baseline system provided by the competition organizers, "based on a simple prompt retrieval approach, derived from SelfCheck-GPT" | [SHROOM participant kit](https://drive.google.com/file/d/1Iv2jKa5XrNfQjzpFnc1WyNtN7AO59W99/view?usp=sharing) |
| v1 | Initial port to LCEL | [shroom_classifier_v1.py](prod/shroom_classifier_v1.py) |
| v1-persona | v1 with personas added | [shroom_classifier_v1_persona.py](prod/shroom_classifier_v1_persona.py) |
| v1-persona-examples | v1 with personas and examples added | [shroom_classifier_v1_persona_examples.py](prod/shroom_classifier_v1_persona_examples.py) |
| v1-conservative | v1 with prompting to err on side of positive classification when rationale is uncertain | [shroom_classifier_v1_conservative.py](prod/shroom_classifier_v1_conservative.py) |
| v2 | Tweaks, reordering of prompts in v1 | [shroom_classifier_v2.py](prod/shroom_classifier_v2.py) |
| v3 | Fina's revised prompt, but with existing majority voting temperature sampling approach to probability estimation | [shroom_classifier_v3.py](prod/shroom_classifier_v3.py) | 
| v3-persona | v3 with personas re-added (one per task) | [shroom_classifier_v3_persona.py](prod/shroom_classifier_v3_persona.py) | 
| v4 | Removed rationale generation | [shroom_classifier_v4.py](prod/shroom_classifier_v4.py) |
| v4-persona | v4 with personas re-added (one per task) | [shroom_classifier_v4_persona.py](prod/shroom_classifier_v4_persona.py) |
| v5 | Increased temperature sampling to 10 | [shroom_classifier_v5.py](prod/shroom_classifier_v5.py) |
| v6 | Added reference instruction (whether to use target, input, or both to determine if output is a hallucination) | [shroom_classifier_v6.py](prod/shroom_classifier_v6.py) |
| v7 | Rearranged prompt to make it possible to insert demonstrations | [shroom_classifier_v7.py](prod/shroom_classifier_v7.py) |
| v8 | Added demonstrations generated using Universal Self-Adaptive Prompting for classification task (K=6, 3 positive, 3 negative)| [shroom_classifier_v8.py](prod/shroom_classifier_v8.py) |
| v9 | v8 with reduced number of demonstrations (K=2, 1 positive, 1 negative) | [shroom_classifier_v9.py](prod/shroom_classifier_v9.py) |

## Runs

| date       | version             | model              |   temperature |   agnostic_acc |   agnostic_rho |   aware_acc |   aware_rho |   avg_acc |   avg_rho |
|:-----------|:--------------------|:-------------------|--------------:|---------------:|---------------:|------------:|------------:|----------:|----------:|
| 2023-12-23 | baseline            | mistral-7b-instruct-v0.2.Q6_K |           0.0   |       0.649299 |       0.380141 |    0.706587 |    0.460958 |  0.677943 |  0.420549 |
| 2024-01-09 | v1                  | gpt-3.5-turbo      |           0.7 |       0.725451 |       0.54904  |    0.712575 |    0.541777 |  0.719013 |  0.545408 |
| 2024-01-09 | v1                  | gpt-4              |           0.7 |       0.801603 |       0.679521 |    0.762475 |    0.555941 |  0.782039 |  0.617731 |
| 2024-01-10 | v1-persona-examples | gpt-3.5-turbo      |           0.7 |       0.671343 |       0.493688 |    0.712575 |    0.524608 |  0.691959 |  0.509148 |
| 2024-01-10 | v1-persona          | gpt-3.5-turbo      |           0.7 |       0.715431 |       0.575568 |    0.720559 |    0.535034 |  0.717995 |  0.555301 |
| 2024-01-11 | v3                  | gpt-3.5-turbo      |           1.2 |       0.735471 |       0.600531 |    0.722555 |    0.516681 |  0.729013 |  0.558606 |
| 2024-01-11 | v2                  | gpt-3.5-turbo      |           0.2 |       0.711423 |       0.488756 |    0.742515 |    0.528502 |  0.726969 |  0.508629 |
| 2024-01-11 | v2                  | gpt-3.5-turbo      |           0.7 |       0.701403 |       0.52644  |    0.742515 |    0.569994 |  0.721959 |  0.548217 |
| 2024-01-11 | v1-conservative     | gpt-3.5-turbo      |           0.7 |       0.701403 |       0.540491 |    0.734531 |    0.568859 |  0.717967 |  0.554675 |
| 2024-01-11 | v3                  | gpt-3.5-turbo      |           0.7 |       0.745491 |       0.584922 |    0.718563 |    0.508341 |  0.732027 |  0.546631 |
| 2024-01-11 | v2                  | gpt-3.5-turbo      |           1.2 |       0.709419 |       0.519809 |    0.752495 |    0.582654 |  0.730957 |  0.551231 |
| 2024-01-12 | v3                  | gpt-4              |           1.2 |       0.821643 |       **0.722481** |    0.782435 |    0.627895 |  **0.802039** |  0.675188 |
| 2024-01-13 | v3                  | gpt-3.5-turbo      |           1.5 |       0.717435 |       0.541716 |    0.708583 |    0.534737 |  0.713009 |  0.538226 |
| 2024-01-13 | v3-persona          | gpt-3.5-turbo      |           1.2 |       0.757515 |       0.612149 |    0.736527 |    0.592091 |  0.747021 |  0.60212  |
| 2024-01-13 | v4                  | gpt-3.5-turbo      |           1.2 |       0.743487 |       0.606295 |    0.748503 |    0.597773 |  0.745995 |  0.602034 |
| 2024-01-14 | v4-persona          | gpt-3.5-turbo      |           1.2 |       0.739479 |       0.585354 |    0.746507 |    0.615651 |  0.742993 |  0.600503 |
| 2024-01-14 | v4-persona          | gpt-4-1106-preview |           1.2 |       0.815631 |       0.7101   |    0.766467 |    0.620185 |  0.791049 |  0.665143 |
| 2024-01-15 | v5                  | gpt-4-1106-preview |           1.2 |       **0.835671** |       0.714804 |    0.762475 |    0.629884 |  0.799073 |  0.672344 |
| 2024-01-16 | v1-persona-examples | gpt-3.5-turbo      |           0.7 |       0.665331 |       0.490591 |    0.720559 |    0.509126 |  0.692945 |  0.499859 |
| 2024-01-16 | v1-persona          | gpt-3.5-turbo      |           0.7 |       0.759519 |       0.630518 |    0.736527 |    0.620753 |  0.748023 |  0.625635 |
| 2024-01-17 | v6                  | gpt-4-1106-preview |           1.2 |       0.817635 |       0.703687 |    0.762475 |    0.623341 |  0.790055 |  0.663514 |
| 2024-01-17 | v7                  | gpt-4-1106-preview |           1.2 |       0.817635 |       0.712696 |    0.764471 |    0.633253 |  0.791053 |  0.672975 |
| 2024-01-20 | v8                  | gpt-4-1106-preview            |           1.2 |       0.779559 |       0.671776 |    **0.786427** |    **0.665739** |  0.782993 |  0.668757 |
| 2024-01-20 | v8                  | gpt-3.5-turbo                 |           1.2 |       0.731463 |       0.591597 |    0.764471 |    0.602794 |  0.747967 |  0.597196 |
| 2024-01-21 | v9                  | gpt-4-1106-preview            |           1.2 |       0.815631 |       0.721391 |    0.778443 |    0.663652 |  0.797037 |  **0.692521** |

## Requirements
- Python 3.11 or higher

## Installation
``$ OPENAI_API_KEY = <your-OpenAI-API-key>``\
``$ pip install -r requirements.txt``

## License
MIT.
