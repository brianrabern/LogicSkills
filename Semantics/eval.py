import requests
from Syntax.convert_to_smt import ast_to_smt2
from Utils.helpers import ast_from_json


# Z3 evaluator
def evaluate(sentence_ast, convert_json=False):

    if convert_json:
        sentence_ast = ast_from_json(sentence_ast)
    print(f"Evaluating sentence AST: {sentence_ast}")
    smt = ast_to_smt2(sentence_ast)["smt2"]
    res = requests.post("http://localhost:8000", data=smt)
    print(f"Z3 response: {res.text.strip()}")
    return res.text.strip()
