evaluation_subject_prompt = """
Your task is to demonstrate that a given argument is invalid by providing a countermodel -- a model in which all the premises are true, but the conclusion is false.
Use the fixed domain [0, 1, 2, 3, 4], and use the following format to provide your countermodel:

- Domain: a list of integers ([0, 1, 2, 3, 4])
- Constants: map each constant to a domain element (e.g., "a": 0)
- Monadic predicates: list of domain elements where the predicate holds (e.g., "F": [0, 2, 3])
- Binary predicates: list of pairs of domain elements (e.g., "R": [[0, 1], [2, 3]])
---
"""
