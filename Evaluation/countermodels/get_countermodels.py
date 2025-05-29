from Database.DB import db
from Database.models import Argument, Sentence
from Semantics.eval import evaluate
import json
import os
from datetime import datetime

session = db.session


def get_argument(argument):
    print("Parsing argument: ", argument.id)
    language = argument.language
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint", language=language).first()
    domain_constraint_ast = domain_constraint.ast

    premise_id_string = argument.premise_ids
    premise_ids = premise_id_string.split(",")
    conclusion_id = argument.conclusion_id

    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    premise_asts = [premise.ast for premise in premises]
    conclusion = session.get(Sentence, conclusion_id)

    joint_ast = premise_asts[0]
    for ast in premise_asts[1:]:
        joint_ast = ["and", joint_ast, ast]
    joint_ast = ["and", joint_ast, domain_constraint_ast]
    argument_ast = ["and", joint_ast, ["not", conclusion.ast]]

    return argument_ast  # in the form: premises+domain_contraint & ~conclusion


def get_countermodel(ast):
    # Use the existing evaluate function to check satisfiability
    model = evaluate(ast, convert_json=True, return_model=True)

    if model:
        print("Countermodel exists - the argument is invalid")
        print("\nCountermodel:")
        print(model)
        return model
    else:
        print("No countermodel exists - the argument is valid")
        return None


def save_countermodel(argument_id, model):
    # Define the file path
    file_path = os.path.join(os.path.dirname(__file__), "countermodels.json")

    # Load existing countermodels if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            countermodels = json.load(f)
    else:
        countermodels = {}

    # Add new countermodel with timestamp
    countermodels[argument_id] = {"model": json.loads(model), "timestamp": datetime.now().isoformat()}

    # Save updated countermodels
    with open(file_path, "w") as f:
        json.dump(countermodels, f, indent=2)


if __name__ == "__main__":

    invalid_arguments = session.query(Argument).filter_by(valid=False).all()

    for argument in invalid_arguments:
        argument_ast = get_argument(argument)
        print("Getting countermodel for argument: ", argument.id)
        model = get_countermodel(argument_ast)
        if model:
            print("Saving countermodel for argument: ", argument.id)
            save_countermodel(argument.id, model)
