from Database.DB import db
from Database.models import Argument, Sentence
from Semantics.eval import evaluate

session = db.session


def get_argument(argument_id):
    argument = session.get(Argument, argument_id)
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
        # return parse_model(model) TODO
    else:
        print("No countermodel exists - the argument is valid")
        return None


# TODO
# def parse_model(model):
#     # parse the model into a dictionary
#     model_dict = {}


# Get the argument and find its countermodel
argument_ast = get_argument("ffff0dcc358c697b")
model = get_countermodel(argument_ast)
