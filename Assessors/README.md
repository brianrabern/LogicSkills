# Assessors

The Assessors module provides the framework for evaluating LLMs on logic tasks extracted from the dataset. It consists of two stages: inference and evaluation. The inference stage gets raw responses from the models and writes them to a JSON file, and the evaluation stage processes and analyzes the responses to assess model performance.

## Overview

It supports evaluating LLMs on three main types of logical reasoning tasks:

1. **Validity Assessment** - Tests whether models can correctly identify valid/invalid logical arguments
2. **Symbolization** - Tests whether models can translate natural language statements into formal logical notation
3. **Countermodel Generation** - Tests whether models can generate counterexamples to invalid arguments

## Architecture

### Core Components

#### `core/` - Shared Infrastructure

- **`llm.py`** - Low-level LLM interface for making API calls (this is where you can tinker to point at local models instead of the third-party API)
- **`response_engine.py`** - Handles inference stage (getting raw responses from models)
- **`evaluation_engine.py`** - Handles evaluation stage (processing and analyzing responses)

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
- **`extras/`** - Not used but a script for getting countermodels in hacky way by hacking Wolfgang's proof generator website <https://github.com/wo/tpg>

## How It Works

### 1. Inference Pipeline

The inference pipeline (`inference_pipeline.py`) handles the first stage:

1. **Configuration Loading** - Loads question type-specific settings from `settings.py`
2. **Question Loading** - Loads questions from JSON files based on language variant
3. **Model Querying** - Uses `ResponseEngine` to query the LLM with each question
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

### Running Inference

I'm just running this from the `inference_pipeline.py` file itself. At the bottom of the file, I have a `if __name__ == "__main__":` block that runs the inference pipeline. And I'm just running the inference pipeline for each question type like this:

```python
# Run inference for validity questions (Carrollian)
pipeline = InferencePipeline("validity", "carroll")
results_file = pipeline.run_pipeline()

# Run inference for symbolization questions (English)
pipeline = InferencePipeline("symbolization", "english")
results_file = pipeline.run_pipeline()

# Run inference for countermodel questions
pipeline = InferencePipeline("countermodel")
results_file = pipeline.run_pipeline()
```

In `settings.py`, I set which model to use for the current run.
There is no doubt a better way to do this, but I'm just doing it this way for now.

### Running Evaluation

Same idea here. I'm just running this from the `evaluation_pipeline.py` file itself. At the bottom of the file, I have a `if __name__ == "__main__":` block that runs the evaluation pipeline. And I'm just manually specifying what inference results to evaluate on the current run. Like this:

```python
# Evaluate validity results
pipeline = EvaluationPipeline("validity", "results/inference/validity/carroll_{model_name}_{timestamp}.json", "openai/gpt-4o-mini")
evaluation_file = pipeline.run_pipeline()

# Evaluate symbolization results
pipeline = EvaluationPipeline("symbolization", "results/inference/symbolization/english_{model_name}_{timestamp}.json", "openai/gpt-4o-mini")
evaluation_file = pipeline.run_pipeline()

# Evaluate countermodel results
pipeline = EvaluationPipeline("countermodel", "results/inference/countermodel/{model_name}_{timestamp}.json", "openai/gpt-4o-mini")
evaluation_file = pipeline.run_pipeline()
```

### Configuration

Question type configurations are defined in `settings.py`:

- Model parameters (temperature, max_tokens, etc.)
- System prompts for each question type
- Output file naming conventions
- Question limits for testing

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
