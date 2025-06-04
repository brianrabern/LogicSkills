def extractor_prompt(raw_response):
    return f"""
# Task
An LLM was given a logic exercise in which it was asked to provide a countermodel to a given argument.
Based on its full response below, extract the countermodel it intended to provide and return it in JSON format.

# Instructions

1. Extract only the countermodel. Do not include any explanatory text.
2. The domain must be a list of integers.
3. Each constant must be assigned to a single integer in the domain.
4. Each monadic predicate must be interpreted as a list of integers from the domain.
5. Each binary predicate must be interpreted as a list of pairs (2-element lists) of integers from the domain.
6. Include all constants and predicates used in the model.
7. The output must be a valid JSON object.

# Output Format

{{
    "Domain": [0, 1, 2, 3, 4],       // the domain of the countermodel
    "b": 1,                          // interpretation of constant 'b'
    "a": 0,                          // interpretation of constant 'a'
    "M": [1],                        // interpretation of the monadic predicate 'M'
    "P": [[1, 1], [1, 2]]            // interpretation of the binary predicate 'P'
}}

# LLM's Raw Response
{raw_response}
"""
