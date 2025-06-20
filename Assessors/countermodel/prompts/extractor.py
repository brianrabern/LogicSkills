def extractor_prompt(raw_response):
    return f"""
# Task
Extract the full countermodel from the LLM’s response below. Return it as a JSON object.

# Requirements
- Only include the countermodel (no explanations or commentary).
- Include **all constants and predicates that appear in the response**, even if empty.
- Types:
  - "Domain": list of integers.
  - Constants: map each constant to a single integer from the domain.
  - Monadic predicates (e.g., M, N, O, F): list of integers from the domain.
  - Binary predicates (e.g., R, Q, P): list of 2-element lists of integers (e.g., [[1, 2], [3, 4]]).
- Output must be valid JSON.

# Example Format
{{
  "Domain": [0, 1, 2, 3, 4],
  "a": 0,
  "b": 1,
  "M": [0, 2],
  "N": [1],
  "O": [4],
  "R": [[1, 4], [4, 1]],
  "P": [[1, 0], [2, 0]]
}}

# LLM's Response
{raw_response}
"""
