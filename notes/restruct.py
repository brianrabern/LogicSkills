import json

# Load the original list
with open("problems_with_ast.json") as f:
    problems_list = json.load(f)

# Convert to dict with ID as key
problems_dict = {
    str(entry["id"]): {k: v for k, v in entry.items() if k != "id"}
    for entry in problems_list
}

# Save to a new file
with open("sentences.json", "w") as f:
    json.dump(problems_dict, f, indent=2, ensure_ascii=False)
