from Database.DB import db
from Database.models import Argument, Sentence
from Utils.helpers import generate_argument_id
import logging
from Utils.logging_config import setup_logging
import time
import json
import random

# Set up logging
log_file = setup_logging("fix_problematic_arguments")
logger = logging.getLogger(__name__)


def load_problematic_arguments(filename):
    """Load problematic arguments from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def get_contradictions(session, language):
    """Get all contradiction sentences (status=-1)."""
    return session.query(Sentence).filter_by(status=-1, language=language).all()


def fix_argument(session, arg_data, contradiction):
    """Fix a problematic argument by replacing its conclusion with a contradiction."""
    try:
        # Get the argument from the database
        arg = session.query(Argument).filter_by(id=arg_data["argument_id"]).first()
        if not arg:
            logger.error(f"Argument {arg_data['argument_id']} not found in database")
            return False

        # Get premise IDs
        premise_ids = arg.premise_ids.split(",")

        # Generate new argument ID with the contradiction as conclusion
        new_id = generate_argument_id(premise_ids, contradiction.id)

        # Update the argument
        old_conclusion_id = arg.conclusion_id
        arg.conclusion_id = contradiction.id
        arg.id = new_id

        session.commit()
        logger.info(f"Fixed argument {arg_data['argument_id']} -> {new_id}")
        logger.info(f"Replaced conclusion {old_conclusion_id} with contradiction {contradiction.id}")
        return True

    except Exception as e:
        logger.error(f"Error fixing argument {arg_data['argument_id']}: {e}")
        session.rollback()
        return False


def main():
    # Load problematic arguments
    filename = "language_mismatched_arguments_20250528_153311.json"
    problematic_args = load_problematic_arguments(filename)
    total_args = len(problematic_args)

    # Get all contradictions
    session = db.session

    # Track statistics
    stats = {"total": total_args, "fixed": 0, "errors": 0, "start_time": time.time()}

    # Fix each problematic argument
    for i, arg_data in enumerate(problematic_args, 1):
        try:
            # Get contradictions in the same language as the argument
            contradictions = get_contradictions(session, arg_data["argument_language"])
            if not contradictions:
                logger.error(f"No contradictions found for language {arg_data['argument_language']}")
                stats["errors"] += 1
                continue

            # Pick a random contradiction
            contradiction = random.choice(contradictions)

            # Fix the argument
            if fix_argument(session, arg_data, contradiction):
                stats["fixed"] += 1
            else:
                stats["errors"] += 1

            # Log progress every 10 arguments
            if i % 10 == 0:
                elapsed = time.time() - stats["start_time"]
                logger.info(f"Progress: {i}/{stats['total']} arguments processed")
                logger.info(f"Fixed: {stats['fixed']}, Errors: {stats['errors']}")
                logger.info(f"Time elapsed: {elapsed:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing argument {arg_data['argument_id']}: {e}")
            stats["errors"] += 1
            continue

    # Final summary
    logger.info("\n=== Final Summary ===")
    logger.info(f"Total problematic arguments: {stats['total']}")
    logger.info(f"Successfully fixed: {stats['fixed']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Total time: {time.time() - stats['start_time']:.2f} seconds")


if __name__ == "__main__":
    main()
