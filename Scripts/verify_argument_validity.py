from Database.DB import db
from Database.models import Argument, Sentence
from Semantics.eval import evaluate
import logging
from Utils.logging_config import setup_logging
import time
import json
from datetime import datetime

# Set up logging
log_file = setup_logging("verify_argument_validity")
logger = logging.getLogger(__name__)


def get_domain_constraint(session, language):
    """Get the domain constraint for a given language."""
    return session.query(Sentence).filter_by(type="domain_constraint", language=language).first()


def validate_argument(session, argument):
    """Validate an argument with domain constraints properly included."""
    try:
        # Get premises and conclusion
        premise_ids = [int(pid) for pid in argument.premise_ids.split(",")]
        premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
        conclusion = session.get(Sentence, argument.conclusion_id)

        # Get domain constraint
        domain_constraint = get_domain_constraint(session, argument.language)
        if not domain_constraint:
            logger.warning(f"No domain constraint found for language {argument.language}")
            return None

        # Build joint AST with premises
        if not premises:
            logger.error(f"No premises found for argument {argument.id}")
            return None

        joint_ast = premises[0].ast
        for premise in premises[1:]:
            joint_ast = ["and", joint_ast, premise.ast]

        # Add domain constraint
        joint_ast = ["and", joint_ast, domain_constraint.ast]

        # Check if conclusion follows
        # We want to check if premises ∧ domain_constraint → conclusion
        # This is equivalent to checking if premises ∧ domain_constraint ∧ ¬conclusion is unsatisfiable
        implication = ["and", joint_ast, ["not", conclusion.ast]]

        result = evaluate(implication, convert_json=True)
        return result == "unsat"

    except Exception as e:
        logger.error(f"Error validating argument {argument.id}: {e}")
        return None


def get_argument_details(session, argument):
    """Get detailed information about an argument for JSON output."""
    premise_ids = [int(pid) for pid in argument.premise_ids.split(",")]
    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    conclusion = session.get(Sentence, argument.conclusion_id)

    return {
        "argument_id": argument.id,
        "language": argument.language,
        "db_valid": argument.valid,
        "premises": [{"id": p.id, "form": p.form, "soa": p.soa} for p in premises],
        "conclusion": {"id": conclusion.id, "form": conclusion.form, "soa": conclusion.soa},
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
        "matches": 0,  # DB and validation match
        "mismatches": 0,  # DB and validation don't match
        "errors": 0,
        "start_time": time.time(),
    }

    # Lists to store mismatched arguments
    mismatched_args = {
        "db_valid_but_invalid": [],  # DB says valid but validation says invalid
        "db_invalid_but_valid": [],  # DB says invalid but validation says valid
    }

    for i, arg in enumerate(arguments, 1):
        try:
            # Validate
            is_valid = validate_argument(session, arg)

            if is_valid is None:
                stats["errors"] += 1
                continue

            # Compare with database
            if is_valid == arg.valid:
                stats["matches"] += 1
            else:
                stats["mismatches"] += 1
                arg_details = get_argument_details(session, arg)
                if arg.valid and not is_valid:
                    mismatched_args["db_valid_but_invalid"].append(arg_details)
                    logger.info(f"Argument {arg.id} is marked valid in DB but validation says invalid")
                else:
                    mismatched_args["db_invalid_but_valid"].append(arg_details)
                    logger.info(f"Argument {arg.id} is marked invalid in DB but validation says valid")

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mismatched_arguments_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(mismatched_args, f, indent=2)
    logger.info(f"Saved mismatched arguments to {filename}")

    # Final summary
    logger.info("\n=== Final Summary ===")
    logger.info(f"Total arguments processed: {stats['total']}")
    logger.info(f"Matches (DB and validation agree): {stats['matches']}")
    logger.info(f"Mismatches (DB and validation disagree): {stats['mismatches']}")
    logger.info(f"  - DB says valid but validation says invalid: {len(mismatched_args['db_valid_but_invalid'])}")
    logger.info(f"  - DB says invalid but validation says valid: {len(mismatched_args['db_invalid_but_valid'])}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Total time: {time.time() - stats['start_time']:.2f} seconds")


if __name__ == "__main__":
    main()
