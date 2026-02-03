# Assessors

The Assessors module provides the framework for evaluating LLMs on LogicSkills. It has two stages: **inference** (get raw model responses and write them to JSON) and **evaluation** (process responses and assess correctness).

## Overview

It supports the three benchmark task types:

1. **Validity assessment** — Decide which conclusion(s), if any, follow from given premises.
2. **Formal symbolization** — Translate sentences into first-order formulas using a given symbol key.
3. **Countermodel construction** — For invalid arguments, provide a finite structure in which all premises are true and the conclusion is false.

## Architecture

### Core Components

#### `core/` - Shared Infrastructure

- **`llm.py`** - Low-level LLM interface for API calls (can be pointed at local models).
- **`prompt_prep.py`** - Builds prompts for inference.
- **`evaluation_engine.py`** - Handles the evaluation stage (querying extractor, parsing responses).

#### Pipeline Components

- **`inference_pipeline.py`** - Orchestrates the inference process for all question types
- **`evaluation_pipeline.py`** - Orchestrates the evaluation process for all question types
- **`settings.py`** - Centralized configuration for all question types and models

### Question Type Modules

Each question type has its own module with specialized components:

#### `validity/`

- **`evaluator.py`** - Evaluates validity assessment responses
- **`generator.py`** - Generates validity assessment questions
- **`prompts/`** - System prompts and extractors for validity tasks
- **`questions_*.json`** - Question datasets (Carrollian and English variants)

#### `symbolization/`

- **`evaluator.py`** - Evaluates symbolization responses with logical equivalence checking
- **`checker.py`** - Checks logical equivalence between formulas
- **`generator.py`** - Generates symbolization questions
- **`prompts/`** - System prompts and extractors for symbolization tasks
- **`questions_*.json`** - Question datasets (Carrollian and English variants)

#### `countermodel/`

- **`evaluator.py`** - Evaluates countermodel generation responses
- **`checker.py`** - Validates countermodels against argument structures
- **`generator.py`** - Generates countermodel questions
- **`prompts/`** - System prompts and extractors for countermodel tasks
- **`questions_*.json`** - Question datasets
- **`extras/`** - Optional scripts for countermodel collection (e.g. using external tools; see [tpg](https://github.com/wo/tpg)).

## How It Works

### 1. Inference Pipeline

The inference pipeline (`inference_pipeline.py`) handles the first stage:

1. **Configuration Loading** - Loads question type-specific settings from `settings.py`
2. **Question Loading** - Loads questions from JSON files based on language variant
3. **Model Querying** - Uses `ModelWrapper` to query the LLM with each question
4. **Result Storage** - Saves raw model responses to timestamped JSON files

### 2. Evaluation Pipeline

The evaluation pipeline (`evaluation_pipeline.py`) handles the second stage:

1. **Result Loading** - Loads inference results from JSON files
2. **Response Evaluation** - Uses specialized evaluators to assess model responses
3. **Answer Extraction** - Extracts structured answers from raw model responses
4. **Correctness Assessment** - Compares extracted answers with ground truth
5. **Result Storage** - Saves evaluation results with accuracy metrics

### 3. Evaluation Process

Each question type has a specialized evaluation process:

- **Validity**: Extracts answer choices and compares with correct answers from the DB
- **Symbolization**: Extracts logical formulas and checks for logical equivalence using Z3
- **Countermodel**: Extracts countermodels and checks for correctness using Z3

## Usage

**Inference** is run from the command line (see root README):

```bash
python Assessors/inference_pipeline.py --model <model> --backend api --question_type validity --language english
```

**Evaluation** is run by setting `CONFIG` at the bottom of `evaluation_pipeline.py` to the desired model config (listing inference JSON paths and `run: True`), then:

```bash
python Assessors/evaluation_pipeline.py
```

For full setup and examples, see the repository root **README.md**.

### Configuration

Question-type settings (prompts, model name, etc.) are in `settings.py`. Model configs (YAML) live in `Models/model_config/`.

## Output Structure

### Inference Results

Stored in `results/inference/{question_type}/`:

```json
{
  "question": {...},
  "response": {
    "raw_response": "model output",
    "inference_metadata": {...},
    "success": true
  }
}
```

### Evaluation Results

Stored in `results/evaluation/{question_type}/`:

```json
{
  "results": [
    {
      "question_id": "...",
      "question_type": "...",
      "model_response": "...",
      "extracted_answer": "...",
      "is_correct": true,
      "comments": {...}
    }
  ],
  "summary": {
    "total_questions": 10,
    "correct_answers": 8,
    "accuracy": 0.8,
    "errors": 0
  }
}
```
