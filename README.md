# JabberBench

A Benchmark for AI Reasoning in a Controlled Nonsense Language

## Overview

JabberBench is designed to rigorously evaluate the deductive reasoning capabilities of large language models. Built on a controlled, English-like syntax populated with Carrollian nonsense words, JabberBench removes semantic cues and world knowledge, forcing models to rely purely on logic. From a small fixed lexicon, millions of grammatical sentences can be compositionally generated, each paired with a corresponding abstract syntax tree. This enables automated, verifiable evaluation using SMT solvers (e.g., Z3 via SMT-LIB 2). By efficiently exploring this structured space, we construct datasets that include both valid deductive arguments and plausible but invalid distractors, yielding a rich resource for testing and training AI models.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/brianrabern/JabberBench.git
   cd JabberBench
   ```

2. Copy the config template and fill in the values:

   ```bash
   cp config.tpl.py config.py
   ```

3. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Set up paths:

   ```bash
   python set_pth.py
   ```

6. Database Setup:
   - Install MariaDB if not already installed
   - Configure the database connection by setting the `mariadb_url` in your `config.py`
   - Create a new database:

     ```sql
     CREATE DATABASE lc_arg;
     ```

   - Load the database from the provided dump file:

     ```bash
     mysql -u root -p lc_arg < lc_arg.sql
     ```

## Usage

There are two main aspects to the project:

1. Generating the database of sentences and arguments (both in English and in the Carrollian language) from which different exercises can be extracted
   - For example, symbolization exercises, countermodel exercises, and validity exercises
2. Evaluating models on these exercises
   - This involves a pipeline of generating questions, prompting the model, extracting the model's responses, and evaluating the model's responses

### Generating the database

A database has already been generated, see `lc_arg.sql`. It was run on an EC2 for a week to generate a large number of arguments.
To make a new database, you essentially run `sentence_generator.py` and `argument_generator.py`. For these to run you need to first have the Z3 server running, see `Semantics/z3_solver.py`.

```bash
python Semantics/z3_solver.py
```

1. In the sentence generator, you can choose which types of sentences to generate. For example, you can generate all sentences, or you can generate only the simple quantified sentences. Then run:

```bash
python Generators/sentence_generator.py
```

Sentences in the Carrollian langauge will be generated along with their English counterparts. This script can be run multiple times to generate more sentences, or to generate sentences of a different type. Sentences that have already been generated will not be generated again.

2. Once there are sentences in the database, you can run the argument generator to find valid (and invalid) arguments:

```bash
python Generators/argument_generator.py
```

This will take a while, but you can stop it and resume it later. There is no danger of adding duplicates to the database, as the argument generator will only add arguments that are not already in the database. When a valid argument is found, the argument generator will also find 5 similar invalid arguments. And it will also add the counterpart English arguments.

### Evaluating models

At present the only evaluation pipeline is for validity exercises. First you need to generate the questions (toggle `num_questions=20` to generate more questions):

```bash
python Evaluation/generate_validity_questions.py
```

This will generate two files: `questions_carroll.json` and `questions_english.json`.

You'll need to inspect `Evaluation/run_evaluation.py` to figure out how to run the evaluation against different models and different question sets.

## Example Exercises

### 1. Argument Validity Exercise

Your task is to solve a logical reasoning problem. Use any approach you find effective, but clearly and explicitly state your final answer.

---

Consider the following situation:

Everything is a tove, or a borogove, or a rath (exclusively), and there's at least one of each. Zindle or Bungo will whiffle. Only toves will whiffle. Every rath chortled at Bungo. If Zindle will whiffle, then every rath is mimsy only if every borogove chortled at Bungo

Which, if any, of the following statements must be true in this situation?

1. Zindle and Bungo will whiffle.
2. Every tove will whiffle.
3. Not all toves will whiffle.
4. A tove will whiffle.
5. No toves will whiffle.
6. Zindle will whiffle.

### 2. Symbolization Exercise

Your task is to translate the provided sentence into formal predicate logic, using the abbreviations and grammar rules provided. Return a single well-formed formula.

---

Sentence:
Every donkey chased Alfred.

Abbreviations:

- M: "[1] is a donkey"
- R: "[1] chased [2]"
- a: "Alfred"

Formal syntax:

WFF        ::= ATOM
            | "¬" WFF
            | "(" WFF CONNECTIVE WFF ")"
            | QUANTIFIER VARIABLE WFF

ATOM       ::= PREDICATE TERM
            | PREDICATE TERM TERM

TERM       ::= VARIABLE | CONSTANT

QUANTIFIER ::= "∀" | "∃"
CONNECTIVE ::= "∧" | "∨" | "→" | "↔"

PREDICATE  ::= A single uppercase letter (A–Z)
VARIABLE   ::= A single lowercase letter from s–z
CONSTANT   ::= A single lowercase letter from a–r

### 3. Countermodel Exercise

Your task is to demonstrate that a given argument is invalid by providing a countermodel -- a model in which all the premises are true, but the conclusion is false. Use the fixed domain [0, 1, 2, 3, 4], and be sure to provide an interpretation for all non-logical symbols. Use the following format:

- Domain: a list of integers ([0, 1, 2, 3, 4])
- Constants: map each constant to a domain element (e.g., "a": 0)
- Monadic predicates: list of domain elements where the predicate holds (e.g., "F": [0, 2, 3])
- Binary predicates: list of pairs of domain elements (e.g., "R": [[0, 1], [2, 3]])

---

Argument:

(∀xFx → ∃x¬Rxb), Rab |= ∀xFx

## Project Structure

### `Database/`

Database models and connection handling

- `DB.py`: Core database connection and session management
- `models.py`: SQLAlchemy models for Arguments, Sentences, and other database entities
- `backups/`: Directory containing database backup files

### `Evaluation/`

Evaluation scripts and tools for testing LLM reasoning

- `run_evaluation.py`: Main script for running evaluations on LLMs
- `evaluator.py`: Core evaluation logic and scoring
- `generate_eval_questions.py`: Generates evaluation questions from the database
- `llm.py`: OpenAI client wrapper for model prompting with OpenRouter integration
- `model.py`: Class for interacting with a model
- `prompts/`: Directory containing evaluation prompts
- `countermodels/`: Tools for generating and verifying countermodels

### `Generators/`

Sentence and argument generators

- `sentence_generator.py`: Generates sentences from the lexicon and adds them to the database
- `argument_generator.py`: Finds arguments (valid and invalid) from the sentences in the database and records them to the database

### `Scripts/`

Collection of inessential scripts for checking, verifying, and fixing the database, etc. (Can be ignored)

### `Semantics/`

Z3 server for evaluating sentence and argument validity

- `z3_solver.py`: HTTP server that provides a REST API for Z3 SMT solving
  - Runs as a standalone service on port 8001
- `eval.py`: Client interface for the Z3 server
  - Converts sentence ASTs to SMT format
  - Communicates with the Z3 server

### `Syntax/`

Language syntax and parsing utilities

- `carroll_lexicon.py`: Lexicon of Carrollian nonsense words
- `english_lexicon.py`: Lexicon of English words
- `parse.py`: Parser for the controlled language
- `convert_to_smt.py`: Converts parsed sentences to SMT-LIB format
- `transform.py`: AST transformation utilities

### `Utils/`

Helper functions and utilities
