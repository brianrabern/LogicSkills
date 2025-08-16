def extractor_prompt(raw_response):
    return f"""
# Task
Extract the full countermodel from the LLM's response below. Return it as a JSON object.

# Requirements
- Only include the countermodel (no explanations or commentary).
- Include **all constants and predicates that appear in the response**, even if empty.
- **IMPORTANT**: Look for symbol assignments in BOTH the explicit definitions AND the reasoning or explanation sections.

# Symbol Types
- **Constants** (a, b, c, etc.): map each constant to a single integer from the domain.
- **Monadic predicates** (M, N, O, K, F, G, etc.): list of integers from the domain that satisfy the predicate.
- **Binary predicates** (P, Q, R, etc.): list of 2-element lists of integers (e.g., [[1, 2], [3, 4]]).

# Extraction Rules
1. **Constants**:
   - Look for patterns like "Let 'a' = 0", "a = 0", "Let a = 0"
   - **ALSO** look for constants mentioned in reasoning or explanation sections like "R(a, b)", "K(a)"
2. **Monadic predicates**: Look for patterns like "M = {{0, 1}}", "M(0) is true", "K(0) is true"
3. **Binary predicates**: Look for patterns like "P = {{(0, 1), (1, 2)}}", "P(0, 1) is true"
4. **Empty predicates**: If a predicate is explicitly set to empty (e.g., "Q = {{}}"), extract as empty list []

# Example Format
{{
  "Domain": [0, 1, 2],
  "a": 0,
  "b": 1,
  "c": 2,
  "M": [0, 2],
  "N": [1],
  "K": [0],
  "R": [[1, 0], [2, 0]],
  "P": [[1, 0], [2, 0]]
}}

# LLM's Response
{raw_response}
"""
