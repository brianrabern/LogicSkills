from Database.DB import db
from Database.models import Argument, Sentence
import random
from Utils.helpers import canonical_premise_str
import json
from pathlib import Path

# Initialize database session
session = db.session


def get_sentence(sentence_id):
    return session.get(Sentence, sentence_id).sentence


def get_invalid_conclusions(premise_ids, valid_conclusion_id, num_options=5):
    # get invalid arguments with the same premises
    premise_str = canonical_premise_str(premise_ids)
    invalid_args = session.query(Argument).filter_by(premise_ids=premise_str, valid=0).all()

    if not invalid_args:
        print(f"Warning: No invalid arguments found for premises {premise_str}")
        return []

    # get the conclusion sentences
    invalid_conclusions = [
        session.get(Sentence, arg.conclusion_id) for arg in invalid_args if arg.conclusion_id != valid_conclusion_id
    ]

    if not invalid_conclusions:
        print(f"Warning: No invalid conclusions found for premises {premise_str}")
        return []

    return random.sample(invalid_conclusions, min(num_options, len(invalid_conclusions)))


def create_question_dict(language, argument):
    # get premises and conclusion
    premise_ids = [int(pid) for pid in argument.premise_ids.split(",")]
    premises = [session.get(Sentence, pid).to_dict() for pid in premise_ids]
    valid_conclusion = session.get(Sentence, argument.conclusion_id)

    # get domain constraint
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint", language=language).first()

    # get invalid options
    invalid_options = get_invalid_conclusions(premise_ids, argument.conclusion_id)
    if not invalid_options:
        print(f"Skipping argument {argument.id} - no invalid options found")
        return None

    # create question text
    question_text = "Consider the following situation:\n\n"

    # add domain constraint if it exists
    if domain_constraint:
        question_text += f"{domain_constraint.sentence} "

    # add premises as a natural paragraph
    premise_texts = [get_sentence(p["id"]) for p in premises]
    question_text += " ".join(premise_texts) + "\n\n"

    question_text += "Which, if any, of the following statements must be true in this situation?\n\n"

    # combine valid and invalid options and shuffle them
    all_options = invalid_options + [valid_conclusion]
    random.shuffle(all_options)

    # create mapping of option numbers to sentence ids
    option_to_sentence_id = {}

    # add options
    for i, option in enumerate(all_options, 1):
        question_text += f"{i}. {option.sentence}\n"
        option_to_sentence_id[str(i)] = option.id

    # get the correct option number
    correct_option = str(all_options.index(valid_conclusion) + 1)

    return {
        "id": argument.id,
        "text": question_text,
        "correct_answer": correct_option,
        "option_to_sentence_id": option_to_sentence_id,
        "premise_ids": premise_ids,
        "domain_constraint_id": domain_constraint.id if domain_constraint else None,
        "language": language,
    }


def generate_eval_questions(language, num_questions=20):
    # get valid arguments
    valid_args = session.query(Argument).filter_by(valid=1, language=language).all()
    selected_args = random.sample(valid_args, min(num_questions, len(valid_args)))

    questions = []
    for arg in selected_args:
        question = create_question_dict(language, arg)
        if question:  # only add if we found invalid options
            questions.append(question)

    return questions


def save_questions(questions, output_file="questions2.json"):
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    try:
        questions = generate_eval_questions(language="carroll", num_questions=20)
        save_questions(questions)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
