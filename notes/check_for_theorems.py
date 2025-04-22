import json
import requests
from New.smt import ast_to_smt2

# Load sentence data
with open("sentences.json") as f:
    sentences = json.load(f)


def ast_from_json(data):
    if isinstance(data, list):
        return tuple(ast_from_json(x) for x in data)
    return data


def get_ast(pid):
    return ast_from_json(sentences[pid]["ast"])


premise_ids = ["7", "158"]
premise_asts = [get_ast(pid) for pid in premise_ids]
premise_ast = ("and", *premise_asts)
conclusion_ast = get_ast("159")


# Z3 entailment checker
def entails(premise_ast, conclusion_ast):
    full_ast = ("and", premise_ast, ("not", conclusion_ast))
    smt = ast_to_smt2(full_ast)["smt2"]
    print(smt)
    res = requests.post("http://localhost:8000", data=smt)
    print(f"Z3 response: {res.text.strip()}")
    return res.text.strip() == "unsat"


# Check entailment
if entails(premise_ast, conclusion_ast):
    print("✅ The conclusion logically follows from the premises.")

# for sentence in sentences:
#     ast = get_ast(sentence)
#     neg_ast = ("not", ast)
#     smt_data = ast_to_smt2(neg_ast)
#     smt_string = smt_data["smt2"]

#     response = requests.post("http://localhost:8000", data=smt_string)
#     if response.text.strip() == "unsat":
#         print("✅ The sentence is valid:", sentence)
