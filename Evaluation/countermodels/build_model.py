import z3
from Database.DB import db
from Utils.helpers import ast_from_json
from Syntax.convert_to_smt import ast_to_smt2


def build_model_assertions(model_dict, domain):
    # Declare constants for domain elements
    domain_consts = {i: z3.Const(f"d{i}", z3.DeclareSort("Object")) for i in domain}

    # Create declarations
    decls = {}

    # Constants
    for name in model_dict["names"]:
        decls[name] = z3.Const(name, domain_consts[0].sort())  # same sort as domain elements

    # Monadic predicates
    for pred in model_dict["monadic_predicates"]:
        decls[pred] = z3.Function(pred, domain_consts[0].sort(), z3.BoolSort())

    # Binary predicates
    for pred in model_dict["binary_predicates"]:
        decls[pred] = z3.Function(pred, domain_consts[0].sort(), domain_consts[0].sort(), z3.BoolSort())

    return decls, domain_consts


def build_model_assertions_from_user_model(user_model, decls, domain_consts):
    assertions = []

    # Assign constants
    for name, val in user_model.items():
        if name == "Domain":  # Skip the Domain key
            continue
        if isinstance(val, int):  # e.g., c = 0
            assertions.append(decls[name] == domain_consts[val])

    # Monadic predicates (e.g., M: [1, 2])
    # Monadic predicates (e.g., M: [1, 2])
    for name, val in user_model.items():
        if name == "Domain":
            continue
        if isinstance(val, list) and all(isinstance(x, int) for x in val):
            for i in domain_consts:
                if i in val:
                    assertions.append(decls[name](domain_consts[i]))
                else:
                    assertions.append(z3.Not(decls[name](domain_consts[i])))

    # Binary predicates (e.g., P: [[0, 1], [0, 2]])
    for name, val in user_model.items():
        if name == "Domain":
            continue
        if isinstance(val, list) and all(isinstance(x, list) and len(x) == 2 for x in val):
            all_pairs = [(i, j) for i in domain_consts for j in domain_consts]
            for i, j in all_pairs:
                if [i, j] in val:
                    assertions.append(decls[name](domain_consts[i], domain_consts[j]))
                else:
                    assertions.append(z3.Not(decls[name](domain_consts[i], domain_consts[j])))

    return assertions


# --- Main model checking logic ---
if __name__ == "__main__":
    # Example user model: Bungo chortled at every tove
    user_model = {
        "Domain": [0, 1, 2],
        "c": 0,  # Bungo is element 0
        "M": [1, 2],  # Toves are elements 1 and 2
        "P": [[0, 1], [0, 2]],  # Bungo chortled at both toves
    }
    sentence = db.get_sentence_where(id=1878)[0]
    print("\nOriginal sentence:", sentence["sentence"])  # Print the actual sentence
    ast = ast_from_json(sentence["ast"])
    print("\nAST:", ast)
    formula_info = ast_to_smt2(("not", ast))
    print("\nSMT2:", formula_info["smt2"])

    domain = user_model["Domain"]
    decls, domain_map = build_model_assertions(formula_info, domain)
    model_assertions = build_model_assertions_from_user_model(user_model, decls, domain_map)

    s = z3.Solver()
    s.add(*model_assertions)

    smt2_lines = formula_info["smt2"].split("\n")
    sort_line = [line for line in smt2_lines if line.startswith("(declare-sort")][0]
    non_decl_lines = [
        line for line in smt2_lines if not line.startswith("(declare-fun") and not line.startswith("(declare-const")
    ]
    smt2_clean = sort_line + "\n" + "\n".join(non_decl_lines[1:])  # skip duplicated sort

    print("\nFinal SMT2 body:")
    print(smt2_clean)
    print("\nZ3 decls:", decls.keys())

    # Now parse the SMT2 string using the existing declarations
    fml = z3.parse_smt2_string(smt2_clean, decls=decls)
    s.add(fml[0])
    print(fml[0])

    print("\nChecking if sentence holds in model...")
    result = s.check()
    print("\nResult:", result)
    if result == z3.sat:
        print("\nSentence is false in the model")
    else:
        print("\nSentence is true in the model")
