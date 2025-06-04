def extractor_prompt(raw_response):
    return f"""
# Task
A multiple-choice question (options 1–6 and "None") has been answered by a language model.
Based on the model's full response, extract its final selected answers.

# Examples

1. Clear single answer:
Input: "After analyzing all options, I conclude that statement 4 must be true because no toves snicker-snacked Bungo."
Output: {{"answer": "4"}}

2. Multiple answers:
Input: "Statements 1, 2, and 3 must be true in this situation because of the first premise."
Output: {{"answer": "1,2,3"}}

3. No correct answers:
Input: "None of the statements must be true in this situation since we have no information about toves."
Output: {{"answer": "None"}}

4. Ambiguous case:
Input: "This is fun. Can you tell me more about toves and borogoves?"
Output: {{"answer": "Indeterminate"}}

# Instructions
1. Extract ONLY the final answer—ignore all intermediate reasoning.
2. The answer must be one of:
   - A single option ("1"–"6") — most common
   - "None" — if the model clearly states that none of the provided statements are true
   - A comma-separated list (e.g., "1,3,5") — only if multiple answers are explicitly selected
   - "Indeterminate" — only if unable to extract any clear answers
3. Your output should be a JSON object with an "answer" field.

# Raw Response
{raw_response}

"""
