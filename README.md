
# JabberBench

A Benchmark for AI Reasoning in a Controlled Nonsense Language

Designed to rigorously evaluate the deductive reasoning capabilities of large language models. Built on a controlled, English-like syntax populated with Carrollian nonsense words, JabberBench removes semantic cues and world knowledge, forcing models to rely purely on logic. From a small fixed lexicon, millions of grammatical sentences can be compositionally generated, each paired with a corresponding abstract syntax tree. This enables automated, verifiable evaluation using SMT solvers (e.g., Z3 via SMT-LIB 2). By efficiently exploring this structured space, we construct datasets that include both valid deductive arguments and plausible but invalid distractors, yielding a rich resource for testing and training AI models.

Evaluation results (small test set )

- meta-llama/llama-3-8b-instruct: accuracy 10%
- meta-llama/llama-3-70b-instruct: accuracy 30%
- mistralai/mixtral-8x7b-instruct: accuracy 25%
- openai/gpt-4o-mini:  accuracy 30%

## how to run

clone it

pip install -r requirements.txt

make config with mariadb_url

make db called lc_arg

create tables using db.create_tables()

python set_pth.py

generate sentences using sentence generator

generate arguments using argument generator
