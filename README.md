# LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models

This repository contains the code and data for **LogicSkills**, a unified benchmark designed to isolate three fundamental skills in formal reasoning: (i) **formal symbolization**—translating premises into first-order logic; (ii) **countermodel construction**—formulating a finite structure in which all premises are true while the conclusion is false; and (iii) **validity assessment**—deciding whether a conclusion follows from a given set of premises.

Items are drawn from the **two-variable fragment of first-order logic (FO2, without identity)**—a decidable logic that supports finite satisfiability while remaining expressive enough to capture nontrivial reasoning. They are presented in both **natural English** and a **Carroll-style language with nonce words**. All examples are **solver-verified** for correctness and non-triviality using the SMT solver **Z3**. LogicSkills uses a **bilingual task design (English and Carrollian)**, with the Carrollian nonce-word condition isolating reasoning from prior semantic knowledge.

## Benchmark overview

The benchmark comprises a **fixed evaluation set of 1,500 problems**:

- **Formal symbolization** — Mapping sentences to logical form; given a sentence and symbol key, output a well-formed formula (600 items: 300 English, 300 Carroll).
- **Countermodel construction** — Demonstrating invalidity via model-theoretic falsification; for an invalid argument, provide a finite structure that makes all premises true and the conclusion false (300 items; language-neutral).
- **Validity assessment** — Recognizing what follows from what; given premises and candidate conclusions, decide which conclusion(s), if any, must follow (600 items: 300 English, 300 Carroll).

Question sets live in `Assessors/{validity,symbolization,countermodel}/questions_*.json`. A normalized JSONL version for each task is in `logicskills/data/` (see `logicskills/README.md`).

## Installation

1. **Clone the repository** (replace with your repo URL if different):

   ```bash
   git clone https://github.com/brianrabern/LogicSkills.git
   cd LogicSkills
   ```

2. **Config** — Copy the template and add your settings (API keys, optional DB URL):

   ```bash
   cp config.tpl.py config.py
   ```

   Edit `config.py`: set `OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`) for API-based inference; set `EXTRACTOR_MODEL` and `JSON_FIXER_MODEL` for evaluation. For database generation only, set `mariadb_url`.

3. **Virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

4. **Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   For local model backends (e.g. vLLM), see `requirements-vllm.txt`.

5. **Paths** (so imports resolve when run from repo root):

   ```bash
   python set_pth.py
   ```

6. **Database (optional)** — Only needed if you want to regenerate sentences/arguments or run the generators:

   - Install MariaDB, set `mariadb_url` in `config.py`, create a database, and load the dump:

   ```bash
   mysql -u root -p -e "CREATE DATABASE lc_arg;"
   mysql -u root -p lc_arg < lc_arg.sql
   ```

## Quick start: run inference and evaluation

You can run the benchmark using the included question sets and the evaluation pipeline without a database.

1. **Start the Z3 server** (needed for symbolization and countermodel evaluation):

   ```bash
   python Semantics/z3_solver.py
   ```

   Keep it running (default: port 8001).

2. **Run inference** — One task and language at a time. Model must match a key in `Models/model_type.py`; use `--backend api` for API models:

   ```bash
   python Assessors/inference_pipeline.py --model openai/gpt-4o --backend api --question_type validity --language english
   python Assessors/inference_pipeline.py --model openai/gpt-4o --backend api --question_type symbolization --language carroll
   python Assessors/inference_pipeline.py --model openai/gpt-4o --backend api --question_type countermodel --language carroll
   ```

   Results are written to `results/inference/{question_type}/` with timestamped filenames.

3. **Run evaluation** — In `Assessors/evaluation_pipeline.py`, set `CONFIG` at the bottom to the model config that lists your inference files (e.g. `gpt_4o`), then:

   ```bash
   python Assessors/evaluation_pipeline.py
   ```

   Evaluation outputs go to `results/evaluation/{question_type}/` and are appended to `results/main_evaluation_results.json`.

## Benchmark data and JSONL export

- **Question sets**: `Assessors/validity/questions_validity_*.json`, `Assessors/symbolization/questions_symbolization_*.json`, `Assessors/countermodel/questions_countermodel.json`.
- **Normalized JSONL** (e.g. for Hugging Face or external tools): run from repo root:

  ```bash
  python Scripts/convert_to_benchmark.py
  ```

  This writes `logicskills/data/symbolization.jsonl`, `validity.jsonl`, and `countermodel.jsonl` from the question sets and system prompts.

  The JSONL files are a **minimal benchmark format** (id, task, input, answer): suitable for running models and for external use (e.g. Hugging Face). To run **this repository’s evaluation pipeline** (Z3 verification, extractor, etc.), you need the **full question sets** in `Assessors/` (the `questions_*.json` files). Those contain the extra fields the evaluators expect (e.g. `argument_ast` for countermodel, `option_to_sentence_id` for validity). Run inference from the question JSONs so that saved results include that structure; then evaluation will work as described above.

## Regenerating data (optional)

If you have the database set up and loaded:

1. **Z3 server** (must be running):

   ```bash
   python Semantics/z3_solver.py
   ```

2. **Sentence generator** — Configure which sentence types to generate in `Generators/sen_generator.py`, then:

   ```bash
   python Generators/sen_generator.py
   ```

   English and Carroll sentences are generated together.

3. **Argument generator** — After sentences exist:

   ```bash
   python Generators/arg_generator.py
   ```

   Finds valid and invalid arguments, plus five invalid distractors per valid argument, and stores them in the database.

## Project structure

| Path | Description |
|------|-------------|
| **Assessors/** | Inference and evaluation pipelines, evaluators, prompts, and question JSONs for validity, symbolization, and countermodel. |
| **Generators/** | `sen_generator.py` (sentences), `arg_generator.py` (arguments); both use the DB and Z3. |
| **Semantics/** | Z3 HTTP server (`z3_solver.py`) and client (`eval.py`) for SMT checking. |
| **Syntax/** | Lexicons (English, Carroll), parser, AST→SMT conversion. |
| **Database/** | SQLAlchemy models and DB connection (optional for running the benchmark). |
| **Models/** | Model configs (YAML in `model_config/`) and wrapper for local/API backends. |
| **Utils/** | Helpers, logging, normalization. |
| **Scripts/** | Utilities including `convert_to_benchmark.py` (builds `logicskills/data/*.jsonl`). |
| **logicskills/** | Dataset card and JSONL data (see `logicskills/README.md`). |
| **results/** | Inference and evaluation outputs (inference JSONs, evaluation JSONs, `main_evaluation_results.json`). |

## License and citation

- **Code**: See the `LICENSE` file in the repository.
- **Benchmark data** (question sets and JSONL in `logicskills/`): CC-BY-4.0.

*This project was originally developed under the codename "JabberBench".*

If you use LogicSkills in your work, please cite the paper:

```bibtex
@misc{rabern2026logicskillsstructuredbenchmarkformal,
      title={LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models}, 
      author={Brian Rabern and Philipp Mondorf and Barbara Plank},
      year={2026},
      eprint={2602.06533},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.06533}, 
}
```


---

*If I had a world of my own, everything would be nonsense. Nothing would be what it is, because everything would be what it isn't. And contrary wise, what is, it wouldn't be. And what it wouldn't be, it would. You see?* — Lewis Carroll
