import random
import multiprocessing as mp
from Database.DB import db
from Database.models import Sentence, Argument
from Utils.helpers import generate_argument_id, canonical_premise_str
from Semantics.eval import evaluate
from sqlalchemy.exc import IntegrityError
import time
import logging
from datetime import datetime
import os
import traceback
from typing import List
from sqlalchemy import func


def process_premises(premises_data):
    """Process a single premise set in a separate process"""
    from Database.DB import db
    from Semantics.eval import evaluate

    try:
        # Create a new session using the sessionmaker
        local_session = db.Session()
        local_evaluator = evaluate
        generator = ArgGenerator(local_session, local_evaluator)

        # Process each premise set in the batch
        for premises in premises_data:
            try:
                # Find arguments using existing logic
                generator.find_arguments(premises)
            except Exception as e:
                logging.error(f"Error processing premises {[p['id'] for p in premises]}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        local_session.close()
    except Exception as e:
        logging.error(f"Critical error in process_premises: {str(e)}")
        logging.error(traceback.format_exc())
        raise


class ArgGenerator:
    def __init__(self, session, evaluator):
        self.session = session
        self.evaluate = evaluator

    def generate_premises(self, sample_size=1000):
        print(f"\nGenerating {sample_size} premise triples...")
        sentences = self.session.query(Sentence).filter(Sentence.status == 0).all()
        print(f"Found {len(sentences)} candidate sentences")
        ids = [s.id for s in sentences]
        for i in range(sample_size):
            if i % 100 == 0:
                print(f"Processed {i} premise triples")
            triple_ids = random.sample(ids, 3)
            triple = self.session.query(Sentence).filter(Sentence.id.in_(triple_ids)).all()
            if len(triple) == 3 and self._share_predicates(triple):
                print(f"Found related premise triple: {[s.id for s in triple]}")
                yield [s.to_dict() for s in triple]

    def _share_predicates(self, triple):
        pred_sets = [set(s.to_dict()["soa"].keys()) for s in triple]
        shared = set.intersection(*pred_sets)
        if len(shared) > 0:
            print(f"Shared predicates: {shared}")
            return True
        return False

    def is_consistent(self, premises):
        print(f"\nChecking consistency of premises: {[p['id'] for p in premises]}")
        domain_constraint = self.session.query(Sentence).filter_by(type="domain_constraint").first()
        joint_ast = [
            "and",
            ["and", premises[0]["ast"], premises[1]["ast"]],
            premises[2]["ast"],
        ]
        if domain_constraint:
            joint_ast = ["and", joint_ast, domain_constraint.ast]
        result = self.evaluate(joint_ast, convert_json=True) == "sat"
        print(f"Premises are {'consistent' if result else 'inconsistent'}")
        return result

    def get_candidate_conclusions(self, premises, limit=1000):
        print(f"\nFinding candidate conclusions for premises: {[p['id'] for p in premises]}")
        symbols = {s for p in premises for s in p["soa"].keys()}
        names = {s for s in symbols if s.islower()}
        preds = {s for s in symbols if s.isupper()}
        print(f"Shared symbols - Names: {names}, Predicates: {preds}")

        # First get a random sample of 5000 sentences
        query = self.session.query(Sentence).filter(Sentence.status == 0, ~Sentence.id.in_([p["id"] for p in premises]))
        total_count = query.count()
        sample_size = min(5000, total_count)
        if sample_size < total_count:
            query = query.order_by(func.random()).limit(sample_size)

        scored = []
        for s in query:
            s_dict = s.to_dict()
            name_overlap = len(names & set(s_dict["soa"].keys()))
            pred_overlap = len(preds & set(s_dict["soa"].keys()))

            # Base score from shared symbols
            score = pred_overlap * 5 + name_overlap * 2

            # Score based on type and subtype
            if s.type == "quantified":
                score += 5  # Base bonus for quantified statements

                # Complex quantified statements get higher scores
                if s.subtype in ["all_all_all", "all_all_back", "some_some_some"]:
                    score += 2  # Triple quantifier patterns
                elif s.subtype in ["rev_some_all", "rev_no_all"]:
                    score += 2  # Reversed quantifier patterns
                elif s.subtype in ["all_all", "some_some_back", "no_some_back"]:
                    score += 2  # Double quantifier patterns
                elif s.subtype in ["all_some", "some_all", "no_all", "no_some"]:
                    score += 1  # Mixed quantifier patterns
            elif s.type == "conditional":
                score -= 10  # Base penalty for conditionals
                if s.subtype == "nested":
                    score -= 2  # Extra penalty for nested conditionals
            elif s.type == "disjunction":
                score -= 2  # Base penalty for disjunctions
            elif s.type == "conjunction":
                if s.subtype == "contrastive":
                    score += 2  # Bonus for contrastive conjunctions
            elif s.type == "biconditional":
                score += 1
            elif s.type == "negation":
                score += 2  # Moderate bonus for negations
            elif s.type == "atomic":
                if s.subtype == "dyadic":
                    score += 1  # Small bonus for dyadic relations

            # Filter out domain constraints
            if s.type == "domain_constraint":
                continue

            # Only include if score is positive
            if score > 0:
                scored.append((score, s_dict))

        scored.sort(reverse=True, key=lambda x: x[0])
        print(f"Found {len(scored)} potential conclusions")
        # # write scored to file
        # scored_dict = [{"score": s[0], "sentence": s[1]} for s in scored[:limit]]
        # with open(f"scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        #     json.dump(scored_dict, f, indent=4)
        return [s for _, s in scored[:limit]]

    def is_valid_argument(self, premises, conclusion):
        print(f"\nValidating argument with conclusion: {conclusion['id']}")

        # First check if conclusion follows from just the domain constraint
        domain_constraint = self.session.query(Sentence).filter_by(type="domain_constraint").first()
        if domain_constraint:
            domain_implication = ["and", domain_constraint.ast, ["not", conclusion["ast"]]]
            if self.evaluate(domain_implication, convert_json=True) == "unsat":
                print("Conclusion follows from domain constraint alone - skipping")
                return False
        # does conclusion follows from domain contraint plus premise 1
        implication = ["and", ["and", domain_constraint.ast, premises[0]["ast"]], ["not", conclusion["ast"]]]
        result = self.evaluate(implication, convert_json=True) == "unsat"
        if result:
            print("Conclusion follows from domain constraint plus premise 1 - skipping")
            return False
        # does conclusion follows from domain contraint plus premise 2
        implication = ["and", ["and", domain_constraint.ast, premises[1]["ast"]], ["not", conclusion["ast"]]]
        result = self.evaluate(implication, convert_json=True) == "unsat"
        if result:
            print("Conclusion follows from domain constraint plus premise 2 - skipping")
            return False
        # does conclusion follows from domain contraint plus premise 3
        implication = ["and", ["and", domain_constraint.ast, premises[2]["ast"]], ["not", conclusion["ast"]]]
        result = self.evaluate(implication, convert_json=True) == "unsat"
        if result:
            print("Conclusion follows from domain constraint plus premise 3 - skipping")
            return False
        # Then check if it follows from premises + domain constraint
        joint_ast = [
            "and",
            ["and", premises[0]["ast"], premises[1]["ast"]],
            premises[2]["ast"],
        ]
        if domain_constraint:
            joint_ast = ["and", joint_ast, domain_constraint.ast]
        implication = ["and", joint_ast, ["not", conclusion["ast"]]]
        result = self.evaluate(implication, convert_json=True) == "unsat"
        print(f"Argument is {'valid' if result else 'invalid'}")
        return result

    def find_arguments(self, premises, limit=1000):
        print(f"\nFinding arguments for premises: {[p['id'] for p in premises]}")
        if not self.is_consistent(premises):
            print("Premises are inconsistent, skipping")
            return None

        premise_ids = [p["id"] for p in premises]
        premise_str = canonical_premise_str(premise_ids)

        MAX_VALID = 5
        MAX_INVALID = 5

        valid_count = self.session.query(Argument).filter_by(premise_ids=premise_str, valid=True).count()
        invalid_count = self.session.query(Argument).filter_by(premise_ids=premise_str, valid=False).count()
        print(f"Current counts - Valid: {valid_count}, Invalid: {invalid_count}")

        candidates = self.get_candidate_conclusions(premises, limit=limit)

        for i, candidate in enumerate(candidates):
            start = time.time()
            if self.is_valid_argument(premises, candidate):
                elapsed = int((time.time() - start) * 1000)
                if valid_count < MAX_VALID:
                    difficulty = self._compute_difficulty_score(
                        premises=premises, conclusion=candidate, search_depth=i, eval_time_ms=elapsed
                    )
                    self.save_argument(premise_ids, candidate["id"], valid=True, difficulty=difficulty)
                    valid_count += 1
                    print(f"Saved valid argument with difficulty {difficulty}")
            else:  # Invalid argument
                if invalid_count < MAX_INVALID:
                    self.save_argument(premise_ids, candidate["id"], valid=False)
                    invalid_count += 1
                    print("Saved invalid argument")

            if valid_count >= MAX_VALID and invalid_count >= MAX_INVALID:
                print("Reached maximum valid and invalid arguments")
                break

        print(f"Final counts - Valid: {valid_count}, Invalid: {invalid_count}")

    def save_argument(self, premise_ids, conclusion_id, valid, difficulty=None, source="z3"):
        premise_str = canonical_premise_str(premise_ids)
        arg_id = generate_argument_id(premise_ids, conclusion_id)

        argument = Argument(
            id=arg_id,
            premise_ids=premise_str,
            conclusion_id=conclusion_id,
            valid=valid,
            difficulty=difficulty,
            source=source,
        )

        try:
            self.session.add(argument)
            self.session.commit()
            print("Successfully saved argument")
        except IntegrityError:
            self.session.rollback()
            print(f"Argument already exists: {arg_id}")

    def _compute_difficulty_score(
        self, premises, conclusion, search_depth, eval_time_ms, w_len=0.2, w_d=0.5, w_t=0.002
    ):
        length_score = sum(len(p["form"]) for p in premises) + len(conclusion["form"])
        return round(w_len * length_score + w_d * search_depth + w_t * eval_time_ms, 2)

    def validate_and_save_argument(self, premises: List[Sentence], conclusion: Sentence, source: str = "manual"):

        try:
            print(f"\nValidating argument with premises: {[p.id for p in premises]}")
            print(f"Conclusion: {conclusion.id}")

            # Convert to dictionaries for consistency checks
            premises_dict = [p.to_dict() for p in premises]
            conclusion_dict = conclusion.to_dict()

            # Check if premises are consistent
            if not self.is_consistent(premises_dict):
                print("Premises are inconsistent")
                return None

            # Validate the argument
            if not self.is_valid_argument(premises_dict, conclusion_dict):
                print("Argument is invalid")
                return None

            # Save the argument
            premise_ids = [p.id for p in premises]
            self.save_argument(premise_ids, conclusion.id, valid=True, difficulty=0, source=source)

            # Return the saved argument
            premise_str = canonical_premise_str(premise_ids)
            return self.session.query(Argument).filter_by(premise_ids=premise_str, conclusion_id=conclusion.id).first()

        except Exception as e:
            print(f"Error validating and saving argument: {str(e)}")
            print(traceback.format_exc())
            return None


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"arg_generation_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file


def generate_parallel(generator, total_premises=10000, batch_size=100, restart_interval=1000):
    """Generate arguments in parallel batches with periodic restarts"""
    log_file = setup_logging()
    logging.info(f"Starting argument generation. Log file: {log_file}")
    logging.info(
        f"Parameters: total_premises={total_premises}, batch_size={batch_size}, restart_interval={restart_interval}"
    )

    processed_premises = 0
    valid_count = 0
    invalid_count = 0
    start_time = datetime.now()
    last_save_time = start_time

    try:
        while processed_premises < total_premises:
            try:
                premise_batches = []
                current_batch = []

                # Collect premise batches for this iteration
                for premises in generator.generate_premises(restart_interval):
                    current_batch.append(premises)
                    if len(current_batch) >= batch_size:
                        premise_batches.append(current_batch)
                        current_batch = []

                if current_batch:
                    premise_batches.append(current_batch)

                # Process batches in parallel
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    pool.map(process_premises, premise_batches)

                # Get current counts and statistics from database
                valid_count = session.query(Argument).filter_by(valid=True).count()
                invalid_count = session.query(Argument).filter_by(valid=False).count()

                # Get difficulty distribution
                difficulties = session.query(Argument.difficulty).filter_by(valid=True).all()
                difficulty_counts = {}
                for d in difficulties:
                    if d[0] is not None:
                        diff_range = int(d[0] // 1)  # Group by integer difficulty
                        difficulty_counts[diff_range] = difficulty_counts.get(diff_range, 0) + 1

                processed_premises += restart_interval
                current_time = datetime.now()
                elapsed_time = (current_time - start_time).total_seconds()
                time_since_last_save = (current_time - last_save_time).total_seconds()

                # Log detailed progress
                logging.info("\n=== Progress Update ===")
                logging.info(f"Processed premises: {processed_premises}/{total_premises}")
                logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
                logging.info(f"Time since last save: {time_since_last_save:.2f} seconds")
                logging.info(f"Valid arguments found: {valid_count}")
                logging.info(f"Invalid arguments found: {invalid_count}")
                logging.info(f"Average time per premise: {elapsed_time/processed_premises:.2f} seconds")
                logging.info("\nDifficulty Distribution:")
                for diff, count in sorted(difficulty_counts.items()):
                    logging.info(f"Difficulty {diff}-{diff+1}: {count} arguments")
                logging.info("=== End Update ===\n")

                last_save_time = current_time
                logging.info("Restarting search...")

            except Exception as e:
                logging.error(f"Error in main generation loop: {str(e)}")
                logging.error(traceback.format_exc())
                # Continue with next iteration
                continue

    except Exception as e:
        logging.error(f"Critical error in generate_parallel: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    finally:
        # Final summary
        total_time = (datetime.now() - start_time).total_seconds()
        logging.info("\n=== Final Summary ===")
        logging.info(f"Total premises processed: {processed_premises}")
        logging.info(f"Total valid arguments: {valid_count}")
        logging.info(f"Total invalid arguments: {invalid_count}")
        logging.info(f"Total time: {total_time:.2f} seconds")
        logging.info(f"Average time per premise: {total_time/processed_premises:.2f} seconds")
        logging.info("=== End Summary ===\n")


if __name__ == "__main__":
    session = db.session
    evaluator = evaluate
    generator = ArgGenerator(session, evaluator)
    generate_parallel(generator, total_premises=10000, batch_size=100, restart_interval=1000)
