from Syntax.parse import parser
from Syntax.transform import transformer
from Syntax.convert_to_smt import ast_to_smt2
import z3
from Assessors.core.evaluation_engine import EvaluationEngine
from config import EXTRACTOR_MODEL


def parse_ast(form):
    """Parse a logical form into an AST."""
    try:
        tree = parser.parse(form)
    except Exception as e:
        print(f"Parse error adding parentheses: {e}")
        # try adding parentheses
        try:
            tree = parser.parse(f"({form})")
        except Exception as e:
            print(f"Trying to fix syntax: {e}")
            try:
                # give to LLM syntax fixer
                model = EvaluationEngine(EXTRACTOR_MODEL)
                response = model.fix_syntax(form)
                # write to file a rcord of attempted syntax fixes
                with open("syntax_fixes.txt", "a") as f:
                    f.write(f"{form} -> {response}\n")
                tree = parser.parse(response)
            except Exception as e:
                print(f"Failed to fix syntax: {e}")
                return None
            return None
    try:
        ast = transformer.transform(tree)
    except Exception as e:
        print(f"Transform error: {e}")
        return None

    return ast


def check_equivalence(model_form, db_form):
    """Check if the model and db ASTs are logically equivalent."""
    model_ast = parse_ast(model_form)
    db_ast = parse_ast(db_form)
    print("model_ast", model_ast)
    print("db_ast", db_ast)
    if model_ast is None:
        # try to repair the formula by adding outer parentheses
        model_ast = parse_ast(f"({model_form})")
        if model_ast is None:
            print("Failed to parse model symbolization with outer parentheses")
            return None
        print("Added outer parentheses to model symbolization")

    neg_biconditional = ast_to_smt2(("not", ("imp", model_ast, db_ast)))
    print("smt2", neg_biconditional["smt2"])
    parsed_formula = z3.parse_smt2_string(neg_biconditional["smt2"])
    print("parsed_formula", parsed_formula)

    solver = z3.Solver()
    solver.add(parsed_formula)
    result = solver.check()
    print("result", result)

    if result == z3.sat:
        print("Model and DB ASTs are not logically equivalent")
        return False
    else:
        print("Model and DB ASTs are logically equivalent")
        return True


if __name__ == "__main__":
    model_form = "∀x(Fx → ¬Gx)"
    db_form = "(¬∃x(Fx ∧ Gx) ∨ (Fa ∧ ¬Fa))"
    check_equivalence(model_form, db_form)
