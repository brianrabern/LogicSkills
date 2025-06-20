evaluation_subject_prompt = """
Your task is to translate the provided sentence into formal predicate logic, using the abbreviations provided.

# Instructions
- Use only the abbreviations given.
- Return a single well-formed formula in standard predicate logic syntax.
- Use standard logical symbols:
  - Quantifiers: ∀, ∃
  - Connectives: ¬, ∧, ∨, →, ↔
- Do **not** include any explanation or extra text—just the formula.

# Example
Sentence: Every linguist admires Charlie.

Abbreviations:
- L: "[1] is a linguist"
- R: "[1] admires [2]"
- c: "Charlie"

Formalization:
∀x(Lx→Rxc)
"""
