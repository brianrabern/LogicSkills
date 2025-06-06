import random
import json
from pathlib import Path
from Database.DB import db
from Database.models import Argument, Sentence
from Utils.normalize import unescape_logical_form
from Utils.helpers import ast_from_json

# Initialize database session
session = db.session


def get_sentence(sentence_id):
    return session.get(Sentence, sentence_id).sentence


def create_question_dict(argument):
    # get the argument form
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint", language=argument.language).first()
    domain_constraint_form = unescape_logical_form(domain_constraint.form)

    premise_id_string = argument.premise_ids
    premise_ids = premise_id_string.split(",")
    conclusion_id = argument.conclusion_id

    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    premises_forms = [unescape_logical_form(premise.form) for premise in premises]

    conclusion = session.get(Sentence, conclusion_id)
    conclusion_form = unescape_logical_form(conclusion.form)

    argument_form = domain_constraint_form
    for premise_form in premises_forms:
        argument_form += ", " + premise_form
    argument_form += " |= " + conclusion_form
    print(argument_form)

    # get argument ast as conditional with premises as antecedent and conclusion as consequent
    domain_contraint_ast = ast_from_json(domain_constraint.ast)
    premises_asts = [ast_from_json(premise.ast) for premise in premises]
    conclusion_ast = ast_from_json(conclusion.ast)

    argument_ast = domain_contraint_ast
    for premise_ast in premises_asts:
        argument_ast = ("and", argument_ast, premise_ast)
    argument_ast = ("imp", argument_ast, conclusion_ast)
    print("\nArgument AST:", argument_ast)

    # create question text
    question_text = "Argument:\n\n" + argument_form

    return {
        "id": argument.id,
        "argument_form": question_text,
        "argument_ast": argument_ast,
    }


def generate_questions(language, num_questions=20):
    # get valid arguments
    invalid_args = (
        session.query(Argument).filter_by(valid=0, language=language).all()
    )  # choose carroll just to aviod getting identical arguments
    selected_args = random.sample(invalid_args, min(num_questions, len(invalid_args)))

    questions = []

    for arg in selected_args:
        question = create_question_dict(arg)
        if question:
            questions.append(question)
    return questions


def save_questions(questions, output_file):
    # save to same directory as this file
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    try:
        language = "Carroll"  # stick to one language so that we don't get identical (counterpart) arguments
        invalid_arguments = generate_questions(language, num_questions=10)
        print(invalid_arguments)

        # Save questions
        save_questions(invalid_arguments, "questions_invalid_arguments.json")
    except Exception as e:
        print(f"Error: {str(e)}")
