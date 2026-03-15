# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHROOM (SemEval-2024 Task 6) is a hallucination detection system that classifies whether language model outputs contain hallucinations across four NLP tasks: Definition Modeling (DM), Paraphrase Generation (PG), Machine Translation (MT), and Text Simplification (TS).

**Core approach:** Prompt an LLM with a task-specific persona, use temperature sampling for N invocations (typically N=10-20), then apply majority voting to estimate p(Hallucination). Evaluation uses accuracy and Spearman's rho correlation on both model-aware and model-agnostic dataset splits.

## Setup

```bash
export OPENAI_API_KEY=<your-key>
pip install -r requirements.txt
```

Requires Python 3.11+.

## Running the Classifier

Classifiers are instantiated and run from Jupyter notebooks in `prod/`:
- `shroom_val.models_run.ipynb` — run on validation data
- `shroom_test.models_run.ipynb` — run on test data
- `shroom_val.models_hyperparameter_study.ipynb` — hyperparameter tuning
- `shroom_val.model_agnostic_ablation_study_metrics.ipynb` — ablation studies

## Scoring

```bash
python3 prod/score.py <submission_dir> <reference_dir> <output_file> [--is_val]
```

Outputs accuracy (`acc`) and Spearman rank correlation (`rho`) per category. Validate submission format with `prod/check_output.py`.

## Architecture

- **`prod/shroom_classifier_v*.py`** — Versioned classifier implementations. Each version is a `ShroomClassifier` class using LangChain (LCEL chains: `ChatPromptTemplate | ChatOpenAI | StrOutputParser`). The latest is v13 with dynamic example selection.
- **`prod/examples.json`** — Pre-generated few-shot examples for in-context prompting.
- **`prod/reference/`** — Train/val/test data splits (`{split}.model-{aware|agnostic}.json`).
- **`prod/results/`** — Classification outputs organized by model/version/date.
- **`prod/scores/`** — Score outputs from evaluation runs.
- **`dev/`** — Experimental classifiers and alternative approaches (ensemble, USP, Likert).

## Classifier Versioning

New versions are created by copying the latest `shroom_classifier_vN.py` and incrementing. Key evolution:
- v1-v4: Prompt engineering (personas, examples, removing rationale)
- v5: Increased temperature sampling (N=10)
- v6-v7: Reference instructions, prompt restructuring for demonstrations
- v8-v9: USAP-generated few-shot examples, tuning example count
- v12: Parameterized `classify()` for ablation (flags: `task_defined`, `role_defined`, `hallucination_defined`, `examples`)
- v13: Dynamic example selection via `examples_per_class` and `example_selection` params

## Datapoint Schema

```json
{
  "id": 0,
  "task": "DM|PG|MT|TS",
  "src": "input text",
  "tgt": "target text",
  "hyp": "generated text",
  "label": "Hallucination|Not Hallucination",
  "p(Hallucination)": 0.7
}
```

## Key Dependencies

LangChain (`langchain`, `langchain_openai`, `langchain_core`) for LLM chain orchestration; `scipy` for Spearman correlation in scoring.
