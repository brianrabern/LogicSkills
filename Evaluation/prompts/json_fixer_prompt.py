def fixer_prompt(broken_output):
    return (
        "The following text was supposed to be a JSON object, but it's invalid, and possibly contains extra text:\n\n"
        f"{broken_output}\n\n"
        "Please fix the formatting and return ONLY the corrected JSON. "
        "Do not include any explanation or markdown formatting."
    )
