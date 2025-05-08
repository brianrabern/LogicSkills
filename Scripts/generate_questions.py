from Database.DB import db
from Database.models import Argument, Sentence
import random
from Utils.normalize import unescape_logical_form
from Utils.helpers import canonical_premise_str

# Initialize database session
session = db.session


def get_english_sentence(sentence_id):
    return session.get(Sentence, sentence_id).sentence


def get_invalid_conclusions(premise_ids, valid_conclusion_id, num_options=5):
    # Get invalid arguments with the same premises
    premise_str = canonical_premise_str(premise_ids)
    invalid_args = (
        session.query(Argument).filter_by(premise_ids=premise_str, valid=0).all()  # valid=0 for invalid arguments
    )

    if not invalid_args:
        print(f"Warning: No invalid arguments found for premises {premise_str}")
        return []

    # Get the conclusion sentences
    invalid_conclusions = [
        session.get(Sentence, arg.conclusion_id)
        for arg in invalid_args
        if arg.conclusion_id != valid_conclusion_id  # Exclude the valid conclusion
    ]

    if not invalid_conclusions:
        print(f"Warning: No invalid conclusions found for premises {premise_str}")
        return []

    # Randomly select num_options
    return random.sample(invalid_conclusions, min(num_options, len(invalid_conclusions)))


def create_question(argument):
    # Get premises and conclusion
    premise_ids = [int(pid) for pid in argument.premise_ids.split(",")]
    premises = [session.get(Sentence, pid).to_dict() for pid in premise_ids]
    valid_conclusion = session.get(Sentence, argument.conclusion_id)

    # Get domain constraint
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint").first()

    # Get invalid options from argument table
    invalid_options = get_invalid_conclusions(premise_ids, argument.conclusion_id)

    if not invalid_options:
        print(f"Skipping argument {argument.id} - no invalid options found")
        return None

    # Create question text
    question = """Consider the following situation:

"""

    # Add domain constraint if it exists
    if domain_constraint:
        question += f"{domain_constraint.sentence} "

    # Add premises as a natural paragraph
    premise_texts = [get_english_sentence(p["id"]) for p in premises]
    question += " ".join(premise_texts) + "\n\n"

    question += """Which, if any, of the following statements must be true in this situation?

"""

    # Combine valid and invalid options and shuffle them
    all_options = invalid_options + [valid_conclusion]
    random.shuffle(all_options)

    # Add options
    for i, option in enumerate(all_options, 1):
        question += f"{i}. {option.sentence}\n"

    # Get the correct option number
    correct_option = all_options.index(valid_conclusion) + 1

    # Get the logical forms for entailment notation
    premise_forms = [unescape_logical_form(session.get(Sentence, pid).form) for pid in premise_ids]
    conclusion_form = unescape_logical_form(valid_conclusion.form)
    domain_form = unescape_logical_form(domain_constraint.form) if domain_constraint else ""

    question += f"""Correct answer: {correct_option}

Logical entailment:
{domain_form + ', ' if domain_form else ''}{', '.join(premise_forms)} |= {conclusion_form}
"""
    question += "=" * 80 + "\n"

    return question


def generate_questions(num_questions=10, specific_argument_id=None):
    # Get valid arguments
    valid_args = session.query(Argument).filter_by(valid=1).all()
    if specific_argument_id:
        valid_args = [arg for arg in valid_args if arg.id == specific_argument_id]
    selected_args = random.sample(valid_args, min(num_questions, len(valid_args)))

    questions = []
    for arg in selected_args:
        question = create_question(arg)
        if question:  # Only add if we found invalid options
            questions.append(question)

    return questions


if __name__ == "__main__":
    try:
        questions = generate_questions(specific_argument_id="3ddddbad4dc1c575")
        for q in questions:
            print(q)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
