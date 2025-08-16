system_prompt = """
# Task
Show that the provided argument is invalid by giving a countermodel -- one where all premises are true and the conclusion is false.

# Instructions
1. You must provide assignments for all constants and predicates used in the argument.
2. Pay attention to the arity of each predicate:
   - Monadic predicates take one argument (e.g., Mx)
   - Binary predicates take two arguments (e.g., Pxy)
3. Use the fixed domain [0, 1, 2]

# Required Format
- Domain: [0, 1, 2]
- Constants: e.g., "a": 0
- Monadic predicates: e.g., "F": [0, 2]
- Binary predicates: e.g., "R": [[0, 1], [2, 0]]
"""
