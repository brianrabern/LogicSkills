import json

# Load the original file
with open("problem_seed.json", "r") as f:
    data = json.load(f)

illfomred = []
ambiguous = []
for d in data:
    if not isinstance(d["form"], list):
        illfomred.append(d)
        print(d)
    if len(d["form"]) != 1:
        ambiguous.append(d)
        print(d)


# Save the results
with open("illfomred.json", "w") as f:
    json.dump(illfomred, f, indent=2, ensure_ascii=False)

with open("ambiguous.json", "w") as f:
    json.dump(ambiguous, f, indent=2, ensure_ascii=False)
