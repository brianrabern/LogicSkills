def normalize_logical_form(form):
    """
    Normalize a logical form by removing whitespace and converting to escaped Unicode.
    """
    # Remove all whitespace
    form = "".join(form.split())

    # Convert to escaped Unicode if not already
    if "\\u" not in form:
        # First encode to bytes with unicode-escape
        form_bytes = form.encode("unicode-escape")
        # Then decode to string and replace \x with \u00
        form = form_bytes.decode("ascii").replace("\\x", "\\u00")

    return form


def unescape_logical_form(form):
    """
    Convert an escaped Unicode string back to its actual Unicode characters.
    Example: '\\u2200x(Ox\\u2192Rxb)' -> '∀x(Ox→Rxb)'
    """
    return form.encode("ascii").decode("unicode-escape")
