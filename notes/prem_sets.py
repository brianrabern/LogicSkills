import itertools
import json

# Load your problem data
with open("sentences.json") as f:
    sentences = json.load(f)

# Get all sentence IDs (keys are strings if JSON loaded)
all_ids = list(sentences.keys())

# Generate all 3-combinations
triples = list(itertools.combinations(all_ids, 3))

# Create indexed dictionary
premise_sets = {
    i + 1: {"premises": list(combo)}  # make sure to convert tuple to list
    for i, combo in enumerate(triples)
}

# Save it
with open("premise_sets.json", "w") as f:
    json.dump(premise_sets, f, indent=2, ensure_ascii=False)
