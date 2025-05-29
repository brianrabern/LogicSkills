from Database.DB import db
from Database.models import Argument, Sentence
import logging
from Utils.logging_config import setup_logging
import time
import json
from datetime import datetime

# Set up logging
log_file = setup_logging("check_argument_languages")
logger = logging.getLogger(__name__)


def get_argument_details(session, argument):
    """Get detailed information about an argument for JSON output."""
    premise_ids = [int(pid) for pid in argument.premise_ids.split(",")]
    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    conclusion = session.get(Sentence, argument.conclusion_id)

    return {
        "argument_id": argument.id,
        "argument_language": argument.language,
        "conclusion_id": conclusion.id,
        "conclusion_language": conclusion.language,
        "premises": [{"id": p.id, "language": p.language} for p in premises],
        "conclusion": {"id": conclusion.id, "language": conclusion.language},
    }


def main():
    session = db.session
    # Get all arguments
    arguments = session.query(Argument).all()
    total_args = len(arguments)
    logger.info(f"Found {total_args} total arguments to verify")

    # Track statistics
    stats = {
        "total": total_args,
        "matches": 0,  # Argument language matches conclusion language
        "mismatches": 0,  # Argument language doesn't match conclusion language
        "errors": 0,
        "start_time": time.time(),
    }

    # List to store mismatched arguments
    mismatched_args = []

    for i, arg in enumerate(arguments, 1):
        try:
            # Get conclusion
            conclusion = session.get(Sentence, arg.conclusion_id)
            if not conclusion:
                logger.error(f"Could not find conclusion {arg.conclusion_id} for argument {arg.id}")
                stats["errors"] += 1
                continue

            # Compare languages
            if arg.language == conclusion.language:
                stats["matches"] += 1
            else:
                stats["mismatches"] += 1
                arg_details = get_argument_details(session, arg)
                mismatched_args.append(arg_details)
                logger.info(
                    f"Argument {arg.id} language mismatch - Argument: {arg.language}, Conclusion: {conclusion.language}"
                )

            # Log progress every 100 arguments
            if i % 100 == 0:
                elapsed = time.time() - stats["start_time"]
                logger.info(f"Progress: {i}/{total_args} arguments processed")
                logger.info(
                    f"Matches: {stats['matches']}, Mismatches: {stats['mismatches']}, Errors: {stats['errors']}"
                )
                logger.info(f"Time elapsed: {elapsed:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing argument {arg.id}: {e}")
            stats["errors"] += 1
            continue

    # Save mismatched arguments to JSON
    if mismatched_args:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"language_mismatched_arguments_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(mismatched_args, f, indent=2)
        logger.info(f"Saved {len(mismatched_args)} mismatched arguments to {filename}")

    # Final summary
    logger.info("\n=== Final Summary ===")
    logger.info(f"Total arguments processed: {stats['total']}")
    logger.info(f"Matches (argument language matches conclusion): {stats['matches']}")
    logger.info(f"Mismatches (argument language differs from conclusion): {stats['mismatches']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Total time: {time.time() - stats['start_time']:.2f} seconds")


if __name__ == "__main__":
    main()
