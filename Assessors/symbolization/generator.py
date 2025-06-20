import random
import json
from pathlib import Path
from Database.DB import db
from Database.models import Sentence
from Utils.normalize import unescape_logical_form


# Initialize database session
session = db.session


def create_question_dict(sentence_info):
    sentence = sentence_info.sentence
    form = sentence_info.form
    abbreviations = sentence_info.soa

    # format abbreviations as a string
    abbreviations_str = "\n".join([f"{k}: {v}" for k, v in abbreviations.items()])

    # create question text
    question_text = "Sentence:\n\n" + sentence + "\n\nAbbreviations:\n\n" + abbreviations_str
    print(question_text)

    return {
        "id": sentence_info.id,
        "question": question_text,
        "form": unescape_logical_form(form),
        "language": sentence_info.language,
    }


def generate_questions(num_questions=20):
    # get Carroll sentences
    sentences = session.query(Sentence).filter_by(language="Carroll").all()
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))

    carroll_questions = []
    english_questions = []

    for sentence in selected_sentences:
        question = create_question_dict(sentence)
        if question:
            carroll_questions.append(question)
            # get English counterpart
            counterpart_id = sentence.counterpart_id
            english_sentence = session.get(Sentence, counterpart_id)
            english_question = create_question_dict(english_sentence)
            if english_question:
                english_questions.append(english_question)

    return carroll_questions, english_questions


def save_questions(questions, output_file):
    # save to same directory as this file
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    try:
        carroll_sentences, english_sentences = generate_questions(num_questions=100)

        # Save questions
        save_questions(carroll_sentences, "questions_symbolization_carroll.json")
        save_questions(english_sentences, "questions_symbolization_english.json")
    except Exception as e:
        print(f"Error: {str(e)}")
