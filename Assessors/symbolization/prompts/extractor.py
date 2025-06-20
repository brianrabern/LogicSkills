def extractor_prompt(raw_response):
    return f"""
# Task
Extract the **well-formed formula (WFF)** from the LLM's response below and return it in the exact JSON format shown.

# Output Format
Return only a JSON object with this structure:
{{ "formula": "..." }}

# Constraints
- The formula must be a **syntactically valid WFF** according to the formal grammar below.
- Do **not** include any explanatory text—only the JSON object.

# Parentheses Rules
- For binary connectives, use: (WFF CONNECTIVE WFF) — include outer parentheses.
- For negation, use: ¬WFF (e.g., ¬Fa, ¬(Fa∧Rab), ¬∀x(Lx→Rxc)) — **do not** wrap inner WFF with parentheses.
- For quantifiers, use: QUANTIFIER VARIABLE WFF (e.g., ∀xLx, ∃yRyc, ∀x(Lx→Rxc))  — **do not** wrap inner WFF with parentheses.
- Do **not** add unnecessary or redundant parentheses

# Formal Grammar
WFF         ::= ATOM
              | "¬" WFF
              | "(" WFF CONNECTIVE WFF ")"
              | QUANTIFIER VARIABLE WFF

ATOM        ::= PREDICATE TERM
              | PREDICATE TERM TERM

TERM        ::= VARIABLE | CONSTANT

QUANTIFIER  ::= "∀" | "∃"
CONNECTIVE  ::= "∧" | "∨" | "→" | "↔"
PREDICATE   ::= A single uppercase letter (A–Z)
VARIABLE    ::= A lowercase letter from **s** to **z**
CONSTANT    ::= A lowercase letter from **a** to **r**

# Example
Input:
"¬(La) & ∀x(L(x) -> R(xc))"

Output:
{{ "formula": "(¬La ∧ ∀x(Lx→Rxc))" }}

# LLM's Response
{raw_response}
"""
