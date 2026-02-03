---
license: cc-by-4.0
pretty_name: LogicSkills
task_categories:
- reasoning
- logic
language:
- en
- art
size_categories:
- 1K<n<10K
---

# LogicSkills

## Dataset Summary

LogicSkills is a benchmark dataset for evaluating formal deductive reasoning in large language models. It is built from a compositional generator that pairs grammatical sentences with formulas in the two-variable fragment of first-order logic (without identity), a decidable logic that supports finite satisfiability while remaining expressive. Each logical form is realized in both controlled English and a Carroll-style nonce-word language, enabling evaluation of reasoning independently of real-world semantic knowledge. The dataset instantiates three task types—validity assessment, formal symbolization, and countermodel construction—each designed to isolate a distinct logical skill. LogicSkills is intended for evaluation and analysis rather than training.

## Tasks

- **Validity assessment**: Determine which candidate conclusion(s), if any, must follow from given premises.

- **Formal symbolization**: Translate a natural-language sentence into a first-order logical formula using a provided symbol key. A canonical formula is provided; correctness is defined up to logical equivalence.

- **Countermodel construction**: Given an invalid argument, construct a finite structure in which all premises are true and the conclusion is false. No reference answers are provided; correctness is verification-based.

## Languages

- English (`en`)

- Carrollian constructed language (`art`)

- Formal first-order logic (used in countermodel prompts)

## Data Format

Each instance uses the following schema:

```yaml
id: unique identifier
task: symbolization | validity | countermodel
language: english | carrollian | formal
input: full prompt shown to the model
answer: task-dependent (canonical formula, label(s), or null)
```

## Dataset Creation

All items were generated automatically and verified for logical correctness and non-triviality using an SMT solver. No human annotation was involved.

## Intended Use

Benchmarking and analysis of deductive reasoning in language models. Not intended for training or fine-tuning.

## Limitations

The dataset is restricted to a controlled logical fragment and uses synthetic language. Symbolization and countermodel tasks require external tooling to verify logical equivalence and semantic satisfaction conditions (e.g., using an SMT solver such as Z3). See the accompanying paper for more details.

The JSONL in this directory is a minimal format (id, task, input, answer) intended for benchmarking and external use. Running the repository’s evaluation scripts (inference + evaluation with Z3 and the extractor) requires the full question sets under `Assessors/`; see the main repository README.

## Ethics

The dataset is fully synthetic and contains no personal or sensitive data.

## Citation

Please cite the accompanying paper if you use this dataset.
