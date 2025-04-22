import json
from transformer import FOLTransformer
from Parsers.parse import parser

transformer = FOLTransformer()


# Make AST JSON-safe
def ast_to_json_compatible(ast):
    if isinstance(ast, tuple):
        return [ast_to_json_compatible(x) for x in ast]
    return ast


# Load dataset
with open("normalized_problems.json") as f:
    problems = json.load(f)

# Add AST to each entry
for entry in problems:
    try:
        form_str = entry["form"][0]
        tree = parser.parse(form_str)
        ast = transformer.transform(tree)
        entry["ast"] = ast_to_json_compatible(ast)
    except Exception as e:
        entry["error"] = f"Parse error: {str(e)}"

# Save enriched file
with open("problems_with_ast.json", "w") as f:
    json.dump(problems, f, indent=2, ensure_ascii=False)
