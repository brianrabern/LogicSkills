import z3
from Database.DB import db
from Utils.helpers import ast_from_json
from Syntax.convert_to_smt import ast_to_smt2


def check_countermodel(user_model, sentence_ast):
    """
    Return True if the user_model is a countermodel for the given sentence.
    """
    comments = []

    # convert sentence AST to SMT (and negate it)
    sentence_info = ast_to_smt2(("not", sentence_ast))
    sentence_smt = sentence_info["smt2"]
    print("\nSentence info:", sentence_info)

    constants = sentence_info.get("names", [])
    monadic = sentence_info.get("monadic_predicates", [])
    binary = sentence_info.get("binary_predicates", [])

    # Validate that all required symbols are interpreted
    for symbol in constants + monadic + binary:
        if symbol not in user_model:
            print(f"Missing interpretation for: {symbol}")
            comments.append(f"Missing interpretation for: {symbol}")
            return False, comments

    # Validate model first
    valid, error = validate_model(user_model)
    if not valid:
        comments.append(error)
        return False, comments

    # convert model to SMT
    smtlib_model = convert_model_to_smtlib(user_model, constants, monadic, binary)
    if smtlib_model is None:
        comments.append("Unable to parse model into SMT")
        return False, comments

    # merge SMT strings and parse
    merged_smt = merge_smts(smtlib_model, sentence_smt)
    print("\nMerged SMT:", merged_smt)
    try:
        parsed_formula = z3.parse_smt2_string(merged_smt)
    except z3.Z3Exception as e:
        print("Z3 parse error:", e)
        comments.append(f"Z3 parse error: {e}")
        return False, comments

    # check satisfiability
    solver = z3.Solver()
    solver.add(parsed_formula)
    result = solver.check()

    # interpret result
    if result == z3.sat:
        print("\nCountermodel:", solver.model())
        return True, comments  # countermodel (sentence is false in this model)
    elif result == z3.unsat:
        comments.append("Not a countermodel")
        return False, comments  # not a countermodel (sentence is true in this model)
    else:
        print("Z3 returned unknown")
        comments.append("Z3 returned unknown")
        return None, comments


def validate_model(model):
    """Validates that a model is well-formed:"""
    if "Domain" not in model:
        return False, "Model does not contain a 'Domain' key"

    domain = model["Domain"]
    if not all(isinstance(x, int) for x in domain):
        return False, "Domain must be a list of integers"

    # Get all values from the model
    all_values = []
    for key, value in model.items():
        if key == "Domain":
            continue
        if isinstance(value, list):
            # Handle both monadic and binary predicates
            if value and isinstance(value[0], list):
                # Binary predicate - flatten pairs
                all_values.extend([x for pair in value for x in pair])
            else:
                # Monadic predicate
                all_values.extend(value)
        else:
            # Constant
            all_values.append(value)

    # Check all values are in domain, using set to get unique values
    invalid_values = sorted(set(v for v in all_values if v not in domain))
    if invalid_values:
        print(f"Model contains values not in domain: {invalid_values}")
        return False, f"Model contains values not in domain: {invalid_values}"
    return True


def convert_model_to_smtlib(model, names, monadic, binary):

    domain = model["Domain"]
    const_map = {i: f"d{i}" for i in domain}
    lines = []

    # Declare domain elements
    for i in domain:
        lines.append(f"(declare-const {const_map[i]} Object)")

    # Declare constants
    for name in names:
        lines.append(f"(declare-const {name} Object)")
        lines.append(f"(assert (= {name} {const_map[model[name]]}))")

    # Monadic predicates
    for pred in monadic:
        lines.append(f"(declare-fun {pred} (Object) Bool)")
        true_set = set(model.get(pred, []))
        for i in domain:
            atom = f"({pred} {const_map[i]})"
            if i in true_set:
                lines.append(f"(assert {atom})")
            else:
                lines.append(f"(assert (not {atom}))")

    # Binary predicates
    for pred in binary:
        lines.append(f"(declare-fun {pred} (Object Object) Bool)")
        true_pairs = set(tuple(p) for p in model.get(pred, []))
        for i in domain:
            for j in domain:
                atom = f"({pred} {const_map[i]} {const_map[j]})"
                if (i, j) in true_pairs:
                    lines.append(f"(assert {atom})")
                else:
                    lines.append(f"(assert (not {atom}))")

    # Domain closure
    or_clauses = " ".join([f"(= x {const_map[i]})" for i in domain])
    lines.append(f"(assert (forall ((x Object)) (or {or_clauses})))")

    return "\n".join(lines)


def extract_declarations(smtlib_str):
    """
    Extracts all (declare-...) lines from an SMT-LIB string.
    Returns a tuple: (list of declarations, list of other lines)
    """
    decls = []
    others = []
    for line in smtlib_str.strip().splitlines():
        line = line.strip()
        if line.startswith("(declare-"):
            decls.append(line)
        elif line:  # ignore empty lines
            others.append(line)
    return decls, others


def merge_smts(model_smt, sentence_smt):
    model_decls, model_rest = extract_declarations(model_smt)
    sent_decls, sent_rest = extract_declarations(sentence_smt)

    # combine and dedupe declarations
    seen = set()
    all_decls = []
    for decl in model_decls + sent_decls:
        if decl not in seen:
            all_decls.append(decl)
            seen.add(decl)

    # combine all parts
    # sort declarations: sort first, then others
    all_decls_sorted = sorted(all_decls, key=lambda d: 0 if "(declare-sort" in d else 1)
    return "\n".join(all_decls_sorted + model_rest + sent_rest)


if __name__ == "__main__":
    from Database.models import Argument, Sentence

    session = db.session

    # get argument
    argument_id = "5614c7ec8cf2a80c"
    argument = session.query(Argument).filter_by(id=argument_id).first()
    print("Parsing argument: ", argument.id)
    language = argument.language
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint", language=language).first()
    premise_id_string = argument.premise_ids
    premise_ids = premise_id_string.split(",")
    conclusion_id = argument.conclusion_id
    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    conclusion = session.get(Sentence, conclusion_id)
    domain_contraint_ast = ast_from_json(domain_constraint.ast)
    premises_asts = [ast_from_json(premise.ast) for premise in premises]
    conclusion_ast = ast_from_json(conclusion.ast)

    argument_ast = domain_contraint_ast
    for premise_ast in premises_asts:
        argument_ast = ("and", argument_ast, premise_ast)
    argument_ast = ("imp", argument_ast, conclusion_ast)

    print("\nArgument AST:", argument_ast)

    user_model = {
        "Domain": [0, 1, 2],
        "c": 0,
        "b": 1,
        "a": 0,
        "M": [1],
        "P": [[1, 1], [1, 2]],
        "N": [0],
        "O": [2],
        "K": [1],
        "L": [2],
    }

    result, comments = check_countermodel(user_model, argument_ast)
    print("\nResult:", result)
    print("\nComments:", comments)

    # #
    # user_model_A = {
    #     "Domain": [0, 1, 2],
    #     "c": 0,
    #     "M": [0, 1, 2],
    #     "P": [[0, 1], [0, 2]],
    # }

    # #
    # user_model_B = {
    #     "Domain": [0, 1, 2],
    #     "c": 3,
    #     "b": 1,
    #     "M": [0, 1, 2],
    #     "P": [[2, 1], [2, 2], [2, 3]],
    # }

    # user_model = user_model_A

    # sentence = db.get_sentence_where(id=1878)[0]
    # print("\nSentence:", sentence["form"])
    # print()

    # sentence_ast = ast_from_json(sentence["ast"])

    # result = check_countermodel(user_model, sentence_ast)
    # print("\nResult:", result)
