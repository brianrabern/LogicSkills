def extractor_prompt(raw_response):
    return f"""
# Task
Extract the **well-formed formula (WFF)** from the LLM's response below. Return it in the specified JSON format.

# Requirements
- Only return the formula as a JSON object: {{ "formula": "..." }}
- The formula must be a WFF
- Be sure to include all parentheses including outer parentheses when offically required
- The formula must match the following syntax:

## Formal Syntax
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
VARIABLE    ::= A lowercase letter from s–z
CONSTANT    ::= A lowercase letter from a–r

# Example
Input:
"La & ∀x(L(x) -> R(xc))"

Output:
{{
  "formula": "(La ∧ ∀x(Lx→Rxc))"
}}

# LLM's Response
{raw_response}
"""
