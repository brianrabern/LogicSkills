from Database.DB import db
from Database.models import Argument, Sentence
import random
from Utils.normalize import unescape_logical_form

# Initialize database session
session = db.session


def get_english_sentence(sentence_id):
    return session.get(Sentence, sentence_id).sentence


def generate_invalid_conclusions(premises, valid_conclusion, num_options=5):
    # Get all sentences that share predicates with the premises
    premise_predicates = set()
    for p in premises:
        premise_predicates.update(p["soa"].keys())

    # Query for potential invalid conclusions
    candidates = (
        session.query(Sentence)
        .filter(Sentence.id != valid_conclusion.id, ~Sentence.id.in_([p["id"] for p in premises]))
        .all()
    )

    # Filter to those sharing some predicates
    candidates = [c for c in candidates if set(c.to_dict()["soa"].keys()) & premise_predicates]

    # Randomly select num_options
    return random.sample(candidates, min(num_options, len(candidates)))


def create_question(argument):
    # Get premises and conclusion
    premise_ids = [int(pid) for pid in argument.premise_ids.split(",")]
    premises = [session.get(Sentence, pid).to_dict() for pid in premise_ids]
    valid_conclusion = session.get(Sentence, argument.conclusion_id)

    # Generate invalid options
    invalid_options = generate_invalid_conclusions(premises, valid_conclusion)

    # Create question text
    question = """Consider the following situation:

"""

    # Add premises as a natural paragraph
    premise_texts = [get_english_sentence(p["id"]) for p in premises]
    question += " ".join(premise_texts) + "\n\n"

    question += """Which of the following statements must be true in this situation?

"""

    # Combine valid and invalid options
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

    question += f"""Correct answer: {correct_option}

Logical entailment:
{', '.join(premise_forms)} |= {conclusion_form}
"""
    question += "=" * 80 + "\n"

    return question


def generate_questions(num_questions=5):
    # Get valid arguments
    valid_args = session.query(Argument).filter_by(valid=True, source="manual").all()
    selected_args = random.sample(valid_args, min(num_questions, len(valid_args)))

    questions = []
    for arg in selected_args:
        questions.append(create_question(arg))

    return questions


if __name__ == "__main__":
    try:
        questions = generate_questions()
        for q in questions:
            print(q)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
