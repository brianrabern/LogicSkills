from collections import defaultdict


def ast_to_smt2(ast):
    """Converts an FOL AST into an SMT-LIB 2 string"""

    def collect_symbols(node, preds=None, names=None):
        if preds is None:
            preds = defaultdict(int)
        if names is None:
            names = set()

        if isinstance(node, tuple):
            tag = node[0]
            if tag == "pred1":
                preds[node[1]] = max(preds[node[1]], 1)
                names.add(node[2])
            elif tag == "pred2":
                preds[node[1]] = max(preds[node[1]], 2)
                names.update([node[2], node[3]])
            elif tag in {"forall", "exists"}:
                names.add(node[1])
                collect_symbols(node[2], preds, names)
            else:  # logic operators
                for child in node[1:]:
                    collect_symbols(child, preds, names)

        return preds, names

    def emit_smt(node):
        if isinstance(node, str):
            return node

        tag = node[0]

        if tag == "pred1":
            return f"({node[1]} {node[2]})"
        elif tag == "pred2":
            return f"({node[1]} {node[2]} {node[3]})"
        elif tag == "not":
            return f"(not {emit_smt(node[1])})"
        elif tag in {"and", "or", "imp", "iff"}:
            op_map = {"and": "and", "or": "or", "imp": "=>", "iff": "="}
            return f"({op_map[tag]} {emit_smt(node[1])} {emit_smt(node[2])})"
        elif tag in {"forall", "exists"}:
            quant = tag
            var = node[1]
            body = emit_smt(node[2])
            return f"({quant} (({var} Object)) {body})"
        else:
            raise ValueError(f"Unknown AST node: {node}")

    # Collect all symbols
    preds, names = collect_symbols(ast)
    print(f"Predicates: {preds}")
    print(f"Names: {names}")

    # Declarations
    decls = ["(declare-sort Object 0)"]

    for p, arity in sorted(preds.items()):
        arg_sorts = " ".join(["Object"] * arity)
        decls.append(f"(declare-fun {p} ({arg_sorts}) Bool)")
    for n in sorted(names):
        decls.append(f"(declare-const {n} Object)")

    # Assert body
    body = emit_smt(ast)
    smt2 = "\n".join(decls + [f"(assert {body})"])

    return {
        "smt2": smt2,
        "names": [
            n
            for n in names
            if n in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]
        ],
        "monadic_predicates": [p for p, arity in preds.items() if arity == 1],
        "binary_predicates": [p for p, arity in preds.items() if arity == 2],
    }
