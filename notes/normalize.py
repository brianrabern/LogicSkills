import json

symbol_map = {
    "names": {
        "a": "Ava",
        "b": "Alfred",
        "c": "Sam",
        "d": "Emily",
        "e": "Ansel",
        "f": "Ruth",
    },
    "predicates": {
        "F1": "[1] studies",
        "F2": "[1] concentrates",
        "F3": "[1] sits",
        "F4": "[1] climbs",
        "F5": "[1] will pass",
        "F6": "[1] will run",
        "F7": "[1] will walk",
        "F8": "[1] will stand",
        "F9": "[1] will bite",
        "F10": "[1] drank",
        "F11": "[1] ate",
        "F12": "[1] sang",
        "F13": "[1] swam",
        "F14": "[1] is mean",
        "F15": "[1] is happy",
        "F16": "[1] is mortal",
        "F17": "[1] is honest",
        "F18": "[1] is sad",
        "F19": "[1] is wet",
        "F20": "[1] is guilty",
        "F21": "[1] is round",
        "F22": "[1] is square",
        "F23": "[1] is an elf",
        "F24": "[1] is a human",
        "F25": "[1] is a knight",
        "F26": "[1] is a knave",
        "F27": "[1] is a monkey",
        "F28": "[1] is a student",
        "F29": "[1] is a person",
        "F30": "[1] is a teacher",
        "F31": "[1] is a number",
        "R1": "[1] loves [2]",
        "R2": "[1] admires [2]",
        "R3": "[1] sees [2]",
        "R4": "[1] kissed [2]",
        "R5": "[1] talked to [2]",
        "R6": "[1] waved to [2]",
        "R7": "[1] is less than [2]",
        "R8": "[1] is greater than [2]",
        "R9": "[1] is a successor of [2]",
    },
}

# Define the normalization function using placeholder substitution


def normalize_entry(entry, symbol_map):
    if "soa" not in entry or "form" not in entry:
        return entry  # skip incomplete entries

    form = entry["form"][0]  # assume single formula

    # Step 1: Create placeholder mapping from original soa keys (F, G, a...) to unique tokens
    placeholder_map = {}
    reverse_map = {}
    for idx, (k, v) in enumerate(entry["soa"].items()):
        placeholder = f"__{idx}__"
        placeholder_map[k] = placeholder
        reverse_map[placeholder] = v

    # Step 2: Replace characters in the form using placeholders
    chars = list(form)
    intermediate_chars = [placeholder_map.get(c, c) for c in chars]
    intermediate_form = "".join(intermediate_chars)

    # Step 3: Final substitution using canonical symbols
    pred_map = {v: k for k, v in symbol_map["predicates"].items()}
    name_map = {v: k for k, v in symbol_map["names"].items()}

    final_form = intermediate_form
    for placeholder, value in reverse_map.items():
        if value in pred_map:
            final_form = final_form.replace(placeholder, pred_map[value])
        elif value in name_map:
            final_form = final_form.replace(placeholder, name_map[value])

    # Step 4: Normalize SOA using canonical symbols
    normalized_soa = {}
    for v in entry["soa"].values():
        if v in pred_map:
            normalized_soa[pred_map[v]] = v
        elif v in name_map:
            normalized_soa[name_map[v]] = v

    # Final update
    entry["form"] = [final_form]
    entry["soa"] = normalized_soa

    return entry


with open("problem_seed.json", "r") as f:
    data = json.load(f)

# Normalize each entry
normalized_data = [normalize_entry(entry, symbol_map) for entry in data]

# Save the results
output_path = "normalized_problems.json"
with open(output_path, "w") as f:
    json.dump(normalized_data, f, indent=2, ensure_ascii=False)

output_path
