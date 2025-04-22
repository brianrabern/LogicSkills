import json
import requests
import random
from Syntax.convert_to_smt import ast_to_smt2


def ast_from_json(data):
    if isinstance(data, list):
        return tuple(ast_from_json(x) for x in data)
    return data


# Load sentence data
with open("sentences.json") as f:
    problems_by_id = json.load(f)

# Load all 3-premise sets
with open("premise_sets.json") as f:
    premise_sets = json.load(f)


# Utility: get the AST from a sentence ID
def get_ast(pid):
    return ast_from_json(problems_by_id[pid]["ast"])


# Z3 entailment checker
def entails(premise_ast, conclusion_ast):
    full_ast = ("and", premise_ast, ("not", conclusion_ast))
    smt = ast_to_smt2(full_ast)["smt2"]
    print(smt)
    res = requests.post("http://localhost:8000", data=smt)
    print(f"Z3 response: {res.text.strip()}")
    return res.text.strip() == "unsat"


# All available sentence IDs
all_ids = set(problems_by_id.keys())

# Loop through premise sets
for key, entry in premise_sets.items():
    premise_ids = entry["premises"]
    try:
        # Compose premise AST
        premise_asts = [get_ast(pid) for pid in premise_ids]
        premise_ast = ("and", *premise_asts)
        all_ids.remove("18")  # remove theorems 18 and 96
        all_ids.remove("96")
        # Candidate conclusions = everything else
        candidate_ids = list(all_ids - set(premise_ids))

        random.shuffle(candidate_ids)

        valid, invalid = [], []

        for cid in candidate_ids:
            if len(valid) >= 5 and len(invalid) >= 5:
                break
            try:
                candidate_ast = get_ast(cid)
                if entails(premise_ast, candidate_ast):
                    if len(valid) < 5:
                        valid.append(cid)
                else:
                    if len(invalid) < 5:
                        invalid.append(cid)
            except Exception as e:
                print(f"Error with {cid} in set {key}: {e}")
                continue

        entry["valid_conclusions"] = valid
        entry["invalid_conclusions"] = invalid
        print(f"✓ Set {key}: valid={valid}, invalid={invalid}")

    except Exception as e:
        entry["error"] = str(e)
        print(f"❌ Error processing set {key}: {e}")

# Save results
with open("problems.json", "w") as f:
    json.dump(premise_sets, f, indent=2, ensure_ascii=False)
