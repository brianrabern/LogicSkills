from Database.DB import db
from Database.models import Sentence
from collections import defaultdict
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_match_key(sentence):
    """Get the key used for matching sentences."""
    return (sentence.form, sentence.type, sentence.subtype)


def match_counterparts():
    """Match Carroll and English sentences and update their counterpart IDs."""
    # Get all sentences
    sentences = db.session.query(Sentence).all()

    # Group sentences by their match key
    groups = defaultdict(list)
    for sentence in sentences:
        groups[get_match_key(sentence)].append(sentence)

    # Track unmatched sentences
    unmatched = []

    # Process each group
    for match_key, group in groups.items():
        carrolls = [s for s in group if s.language == "carroll"]
        englishes = [s for s in group if s.language == "english"]

        if len(carrolls) == 1 and len(englishes) == 1:
            # Perfect match - update counterpart IDs
            carroll = carrolls[0]
            english = englishes[0]

            carroll.counterpart_id = english.id
            english.counterpart_id = carroll.id

            logger.info(f"Matched: Carroll ID {carroll.id} with English ID {english.id}")
            logger.info(f"Form: {carroll.form}")
            logger.info(f"Type: {carroll.type}, Subtype: {carroll.subtype}")
            logger.info("-" * 50)
        else:
            # Log unmatched or ambiguous cases
            for carroll in carrolls:
                unmatched.append(
                    {
                        "id": carroll.id,
                        "form": carroll.form,
                        "type": carroll.type,
                        "subtype": carroll.subtype,
                        "sentence": carroll.sentence,
                        "reason": "multiple_matches" if len(englishes) > 1 else "no_match",
                    }
                )

    # Save unmatched sentences to JSON
    if unmatched:
        output_path = Path("unmatched_sentences.json")
        with open(output_path, "w") as f:
            json.dump(unmatched, f, indent=2)
        logger.info(f"Saved {len(unmatched)} unmatched sentences to {output_path}")

    # Commit changes to database
    try:
        db.session.commit()
        logger.info("Successfully committed all changes to database")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error committing changes: {e}")


if __name__ == "__main__":
    match_counterparts()
