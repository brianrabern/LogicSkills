import random
from Database.DB import db
from Database.models import Sentence, Argument
from Utils.helpers import generate_argument_id, canonical_premise_str
from Semantics.eval import evaluate
from sqlalchemy.exc import IntegrityError
import time
import logging
from datetime import datetime
import traceback
from typing import List
from Utils.logging_config import setup_logging
import math
import itertools
from cachetools import TTLCache
import json
import os
import threading


class ArgGenerator:
    """Generates logical arguments from a database of sentences."""

    def __init__(self, session, evaluator, lexicon, num_premises=3, subset_percentage=0.2):
        """Initialize the argument generator."""
        self.session = session
        self.evaluate = evaluator
        self.num_premises = num_premises
        self.subset_percentage = subset_percentage
        self.lexicon = lexicon
        self.language = lexicon.language

        # Initialize caches with 1-hour TTL
        self._premise_ast_cache = TTLCache(maxsize=1000, ttl=3600)
        self._conclusion_from_subset_cache = TTLCache(maxsize=1000, ttl=3600)

        # Define predicate groups for coherent story generation
        self.kind_predicates = {
            X for X in self.lexicon.predicates.keys() if self.lexicon.predicates[X]["semantic_type"] == "kind"
        }
        self.action_predicates = {
            X for X in self.lexicon.predicates.keys() if self.lexicon.predicates[X]["semantic_type"] == "action"
        }
        self.state_predicates = {
            X for X in self.lexicon.predicates.keys() if self.lexicon.predicates[X]["semantic_type"] == "state"
        }
        self.names = {X for X in self.lexicon.names.keys()}

        self._domain_constraint = None
        self._domain_constraint_ast = None

    def _get_domain_constraint(self):
        """Get and cache the domain constraint sentence and its AST."""
        try:
            if self._domain_constraint is None:
                self._domain_constraint = (
                    self.session.query(Sentence).filter_by(type="domain_constraint", language=self.language).first()
                )
                if self._domain_constraint:
                    self._domain_constraint_ast = self._domain_constraint.ast
                    logging.info(f"Loaded domain constraint with ID: {self._domain_constraint.id}")
                else:
                    logging.warning("No domain constraint found in database")
            return self._domain_constraint
        except Exception as e:
            logging.error(f"Error loading domain constraint: {e}")
            return None

    def _get_premise_ast(self, premise_id):
        """Get and cache premise AST."""
        try:
            if premise_id not in self._premise_ast_cache:
                premise = self.session.get(Sentence, premise_id)
                if premise:
                    self._premise_ast_cache[premise_id] = premise.ast
                else:
                    logging.warning(f"Premise {premise_id} not found")
            return self._premise_ast_cache[premise_id]
        except Exception as e:
            logging.error(f"Error getting AST for premise {premise_id}: {e}")
            return None

    def _generate_coherent_predicate_set(self):
        """Generate a coherent set of predicates and names for story generation."""
        # Select all kind predicates
        selected_kinds = self.kind_predicates
        logging.debug(f"Selected kind predicates: {selected_kinds}")

        # Select 4-5 action/state predicates
        num_actions = random.randint(4, 5)
        action_states = self.action_predicates | self.state_predicates
        selected_actions = set(random.sample(list(action_states), num_actions))
        logging.debug(f"Selected {num_actions} action/state predicates: {selected_actions}")

        # Select all names
        selected_names = self.names
        logging.debug(f"Selected names: {selected_names}")

        return selected_kinds | selected_actions, selected_names

    def generate_premises(self, sample_size=1000):
        """Generate premise sets from the database."""
        logging.info(f"Generating {sample_size} premise sets of size {self.num_premises}")

        try:
            # Count available sentences for this language
            total_sentences = (
                self.session.query(Sentence.id).filter(Sentence.status == 0, Sentence.language == self.language).count()
            )
            logging.info(f"Found {total_sentences} candidate sentences for language {self.language}")

            # Generate random IDs in batches
            batch_size = 1000
            for i in range(0, sample_size, batch_size):
                current_batch = min(batch_size, sample_size - i)
                if i % 1000 == 0:
                    logging.info(f"Processed {i} premise sets")

                # Generate a coherent predicate set for this batch
                selected_predicates, selected_names = self._generate_coherent_predicate_set()
                logging.debug(f"Generated coherent predicate set: {selected_predicates}")
                logging.debug(f"Selected names: {selected_names}")

                try:
                    # Get random IDs for this language
                    ids = random.sample(range(1, total_sentences + 1), current_batch * self.num_premises)

                    # Fetch sentences in batches
                    for j in range(0, len(ids), self.num_premises):
                        premise_ids = ids[j : j + self.num_premises]
                        premises = (
                            self.session.query(Sentence)
                            .filter(Sentence.id.in_(premise_ids), Sentence.language == self.language)
                            .all()
                        )

                        # Check if premises use only the selected predicates and names
                        valid_premises = []
                        rejected_premises = []
                        for p in premises:
                            p_dict = p.to_dict()
                            # Properly separate predicates (uppercase) from names (lowercase)
                            p_predicates = {s for s in p_dict["soa"].keys() if s.isupper()}
                            p_names = {s for s in p_dict["soa"].keys() if s.islower()}

                            # Check if premise's SOA is a subset of our selected set
                            if p_predicates.issubset(selected_predicates) and p_names.issubset(selected_names):
                                valid_premises.append(p)
                            else:
                                rejected_premises.append((p.id, p_predicates, p_names))

                        if len(valid_premises) == self.num_premises:
                            logging.info(f"Found related premise set: {[s.id for s in valid_premises]}")
                            yield [s.to_dict() for s in valid_premises]
                        elif len(rejected_premises) > 0 and i % 1000 == 0:  # Log some rejections for debugging
                            logging.debug("\nSample of rejected premises:")
                            for p_id, preds, names in rejected_premises[:3]:  # Show first 3 rejections
                                logging.debug(f"ID: {p_id}")
                                logging.debug(f"Predicates: {preds}")
                                logging.debug(f"Names: {names}")
                                logging.debug(f"Extra predicates: {preds - selected_predicates}")
                                logging.debug(f"Extra names: {names - selected_names}")

                except Exception as e:
                    logging.error(f"Error processing batch {i}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error in generate_premises: {e}")
            raise

    def _share_predicates(self, premises):
        """Check if the premises share any predicates."""
        try:
            pred_sets = [set(s.to_dict()["soa"].keys()) for s in premises]
            shared = set.intersection(*pred_sets)
            if len(shared) > 0:
                logging.debug(f"Found shared predicates: {shared}")
                return True
            logging.debug("No shared predicates found")
            return False
        except Exception as e:
            logging.error(f"Error checking shared predicates: {e}")
            return False

    def is_consistent(self, premises):
        """Check if the premises are logically consistent."""

        logging.info(f"Checking consistency of premises: {[p['id'] for p in premises]}")
        try:
            # Get domain constraint
            domain_constraint = (
                self.session.query(Sentence).filter_by(type="domain_constraint", language=self.language).first()
            )
            if domain_constraint:
                logging.debug("Including domain constraint in consistency check")

            # Build joint AST for all premises
            try:
                joint_ast = premises[0]["ast"]
                for premise in premises[1:]:
                    joint_ast = ["and", joint_ast, premise["ast"]]

                if domain_constraint:
                    joint_ast = ["and", joint_ast, domain_constraint.ast]
            except (KeyError, IndexError) as e:
                logging.error(f"Error building joint AST: {e}")
                return False

            # Check satisfiability
            try:
                result = self.evaluate(joint_ast, convert_json=True) == "sat"
                logging.info(f"Premises are {'consistent' if result else 'inconsistent'}")
                return result
            except Exception as e:
                logging.error(f"Error evaluating consistency: {e}")
                return False

        except Exception as e:
            logging.error(f"Error checking consistency: {e}")
            return False

    def _conclusion_follows_from_subset(self, subset, conclusion_ast):
        """Check if the conclusion follows from a subset of premises."""
        try:
            # Get ASTs for all premises in subset
            subset_asts = []
            for p in subset:
                ast = self._get_premise_ast(p["id"])
                if ast is None:
                    logging.error(f"Failed to get AST for premise {p['id']}")
                    return False
                subset_asts.append(ast)

            # Build joint AST
            try:
                joint_ast = subset_asts[0]
                for ast in subset_asts[1:]:
                    joint_ast = ["and", joint_ast, ast]
            except IndexError as e:
                logging.error(f"Error building joint AST: {e}")
                return False

            # Check if conclusion follows
            try:
                implication = ["and", joint_ast, ["not", conclusion_ast]]
                result = self.evaluate(implication, convert_json=True) == "unsat"
                logging.debug(f"Conclusion {'follows' if result else 'does not follow'} from subset")
                return result
            except Exception as e:
                logging.error(f"Error evaluating implication: {e}")
                return False

        except Exception as e:
            logging.error(f"Error checking conclusion follows from subset: {e}")
            return False

    def _check_domain_constraints(self, conclusion):
        """Check if the conclusion follows from domain constraints alone."""

        logging.info(f"Checking if conclusion {conclusion['id']} follows from domain constraints")

        try:
            # Get cached domain constraint
            domain_constraint = self._get_domain_constraint()
            if not domain_constraint:
                logging.debug("No domain constraint found")
                return False

            # Get conclusion AST
            try:
                conclusion_ast = conclusion["ast"]
            except KeyError as e:
                logging.error(f"Error accessing conclusion AST: {e}")
                return False

            # Check if conclusion follows from domain constraint
            try:
                domain_implication = ["and", self._domain_constraint_ast, ["not", conclusion_ast]]
                result = self.evaluate(domain_implication, convert_json=True) == "unsat"
                if result:
                    logging.info("Conclusion follows from domain constraint alone")
                return result
            except Exception as e:
                logging.error(f"Error evaluating domain constraint implication: {e}")
                return False

        except Exception as e:
            logging.error(f"Error checking domain constraints: {e}")
            return False

    def is_keeper_argument(self, premises, conclusion):
        """Check if the argument is a keeper (valid and not too trivial).

        This method performs a two-stage validation:
        1. First checks if the conclusion follows from any subset of premises of size subset_percentage
        2. Then checks if the argument is valid

        Returns:
        - True: Argument is a keeper (valid and not too trivial)
        - False: Argument is invalid
        - None: Either validation failed due to error, or argument was skipped because too trivial
        """
        try:
            # Stage 0: Check domain constraints
            if self._check_domain_constraints(conclusion):
                logging.info("Skipping argument - conclusion follows from domain constraints alone")
                return None

            # Get conclusion AST
            conclusion_ast = self._get_premise_ast(conclusion["id"])
            if not conclusion_ast:
                logging.error("Failed to get conclusion AST")
                return None

            # Stage 1: Check subsets
            if self._check_premise_subsets(premises, conclusion["id"], conclusion_ast):
                return None

            # Stage 2: Check all premises
            return self._is_valid_argument(premises, conclusion_ast)

        except Exception as e:
            logging.error(f"Error in is_keeper_argument: {e}")
            return None

    def _is_valid_argument(self, premises, conclusion_ast):
        """Check if conclusion follows from all premises together."""

        try:
            premise_asts = [self._get_premise_ast(p["id"]) for p in premises]
            if not all(premise_asts):
                logging.error("Failed to get all premise ASTs")
                return None

            joint_ast = premise_asts[0]
            for ast in premise_asts[1:]:
                joint_ast = ["and", joint_ast, ast]
            implication = ["and", joint_ast, ["not", conclusion_ast]]

            result = self.evaluate(implication, convert_json=True) == "unsat"
            if result:
                logging.info(f"Found keeper argument requiring all premises: {[p['id'] for p in premises]}")
            else:
                logging.info("Argument is invalid - conclusion doesn't follow from all premises")
            return result

        except Exception as e:
            logging.error(f"Error in full validation: {e}")
            return None

    def _record_valid_subset(self, premise_ids, conclusion_id):
        """Record a valid premise subset and its conclusion to a JSON file."""
        try:
            # Create directory for subset records if it doesn't exist
            subset_dir = "subset_records"
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir)

            # File to store subset records
            subset_file = os.path.join(subset_dir, "valid_subsets.json")

            # Load existing records if file exists
            existing_records = []
            if os.path.exists(subset_file):
                try:
                    with open(subset_file, "r") as f:
                        existing_records = json.load(f)
                except json.JSONDecodeError:
                    logging.warning("Could not parse existing subset records file, starting fresh")

            # Create new record
            record = {
                "premise_ids": premise_ids,
                "conclusion_id": conclusion_id,
                "timestamp": datetime.now().isoformat(),
            }
            existing_records.append(record)

            # Save updated records
            with open(subset_file, "w") as f:
                json.dump(existing_records, f, indent=2)
            logging.info(f"Recorded valid subset in {subset_file}")

        except Exception as e:
            logging.error(f"Error recording valid subset: {e}")

    def _check_premise_subsets(self, premises, conclusion_id, conclusion_ast):
        """Check if conclusion follows from any subset of premises."""

        subset_size = math.ceil(len(premises) * self.subset_percentage)
        logging.info(f"Checking {len(premises)} premises with subset size {subset_size}")

        for subset in itertools.combinations(premises, subset_size):
            # Include conclusion ID in cache key
            subset_key = (conclusion_id, tuple(sorted(p["id"] for p in subset)))
            subset_ids = [p["id"] for p in subset]

            # Check cache first
            if subset_key in self._conclusion_from_subset_cache:
                if self._conclusion_from_subset_cache[subset_key]:
                    logging.info(f"Found cached result - conclusion {conclusion_id} follows from subset: {subset_ids}")
                    return True
                continue

            # Check if conclusion follows from this subset
            try:
                if self._conclusion_follows_from_subset(subset, conclusion_ast):
                    self._conclusion_from_subset_cache[subset_key] = True
                    logging.info(f"Conclusion {conclusion_id} follows from subset: {subset_ids}")

                    # Record the valid subset
                    if len(subset_ids) > 2:
                        self._record_valid_subset(subset_ids, conclusion_id)
                    return True

                self._conclusion_from_subset_cache[subset_key] = False
            except Exception as e:
                logging.error(f"Error checking subset {subset_ids}: {e}")
                continue

        logging.info(f"No valid subsets found for conclusion {conclusion_id}")
        return False

    def get_candidate_conclusions(self, premises, limit=1000):
        """Find potential conclusions for the given premises."""
        logging.info(f"Finding candidate conclusions for premises: {[p['id'] for p in premises]}")

        # Extract symbols from premises
        symbols = {s for p in premises for s in p["soa"].keys()}
        names = {s for s in symbols if s.islower()}
        preds = {s for s in symbols if s.isupper()}
        logging.debug(f"Available symbols - Names: {names}, Predicates: {preds}")

        # Get a random sample of sentences, excluding conditionals, nested conditionals, and disjunctions
        premise_ids = [p["id"] for p in premises]
        query = self.session.query(Sentence).filter(
            Sentence.status == 0,
            Sentence.language == self.language,
            ~Sentence.id.in_(premise_ids),
            ~Sentence.type.in_(["conditional", "disjunction"]),  # Exclude conditionals and disjunctions
            ~Sentence.subtype.in_(["nested"]),  # Exclude nested conditionals
        )

        total_count = query.count()
        sample_size = min(10000, total_count)
        if sample_size < total_count:
            offset = random.randint(0, total_count - sample_size)
            query = query.offset(offset).limit(sample_size)
        logging.debug(f"Sampling {sample_size} sentences from {total_count} total")

        # Filter candidates that only use available symbols
        candidates = []
        for s in query:
            s_dict = s.to_dict()
            s_symbols = set(s_dict["soa"].keys())
            s_names = {s for s in s_symbols if s.islower()}
            s_preds = {s for s in s_symbols if s.isupper()}

            # Only keep sentences that use a subset of our available symbols
            if s_preds.issubset(preds) and s_names.issubset(names):
                candidates.append(s_dict)
                logging.debug(f"Found candidate {s_dict['id']} using symbols: {s_symbols}")

        logging.info(f"Found {len(candidates)} potential conclusions")
        return candidates[:limit]

    def get_invalid_candidates(self, valid_conclusion, limit=5000):
        """Find potential invalid conclusions that match the valid conclusion's characteristics."""
        valid_type = valid_conclusion["type"]
        valid_symbols = set(valid_conclusion["soa"].keys())

        logging.info(f"Finding invalid candidates matching type: {valid_type}")
        logging.debug(f"Valid conclusion symbols: {valid_symbols}")

        # Get candidates of same type
        same_type_matches = self._get_matching_candidates(
            valid_type=valid_type, valid_symbols=valid_symbols, valid_id=valid_conclusion["id"], same_type=True
        )

        # Get candidates of different type
        diff_type_matches = self._get_matching_candidates(
            valid_type=valid_type, valid_symbols=valid_symbols, valid_id=valid_conclusion["id"], same_type=False
        )

        # Return same type matches first, then different type matches
        candidates = same_type_matches + diff_type_matches
        logging.info(
            f"Found {len(same_type_matches)} same type matches and {len(diff_type_matches)} different type matches"
        )
        return candidates[:limit]

    def _get_matching_candidates(self, valid_type, valid_symbols, valid_id, same_type=True):
        """Get candidates that match the valid conclusion's characteristics."""
        # Build query
        query = self.session.query(Sentence).filter(Sentence.status == 0)
        if same_type:
            query = query.filter(Sentence.type == valid_type)
        else:
            query = query.filter(Sentence.type != valid_type)

        # Get count for logging
        count = query.count()
        logging.debug(f"Found {count} sentences of {'same' if same_type else 'different'} type")

        # Find matches
        matches = []
        for s in query:
            s_dict = s.to_dict()
            if s_dict["id"] == valid_id:
                continue
            if set(s_dict["soa"].keys()) == valid_symbols:
                logging.debug(f"Found {('same' if same_type else 'different')} type match: {s_dict['id']}")
                matches.append(s_dict)

        return matches

    def find_invalid_arguments(self, premises, valid_conclusion, candidates=None, max_invalid=5):
        """Find invalid arguments that match the characteristics of a valid argument."""
        logging.info(f"Finding invalid arguments for valid conclusion {valid_conclusion['id']}")

        # Get candidates that match the valid conclusion's characteristics
        invalid_candidates = self.get_invalid_candidates(valid_conclusion, limit=1000)
        if candidates:
            invalid_candidates.extend(candidates)
        logging.info(f"Found {len(invalid_candidates)} potential invalid candidates")

        invalid_count = 0
        premise_ids = [p["id"] for p in premises]

        for candidate in invalid_candidates:
            # Skip if we've found enough invalid arguments
            if invalid_count >= max_invalid:
                logging.info(f"Reached maximum of {max_invalid} invalid arguments")
                break

            # Check if it's actually invalid
            is_valid = self.is_keeper_argument(premises, candidate)
            if is_valid is None:
                logging.debug(f"Skipping candidate {candidate['id']} - validation failed")
                continue

            if not is_valid:
                # Save explicitly invalid arguments
                self.save_argument(premise_ids, candidate["id"], valid=False)
                invalid_count += 1
                logging.info(f"Saved invalid argument {candidate['id']} with matching characteristics")

        logging.info(f"Found and saved {invalid_count} invalid arguments")
        return invalid_count

    def find_arguments(self, premises, limit=1000):
        """Find valid and invalid arguments for the given premises."""
        logging.info(f"Finding arguments for premises: {[p['id'] for p in premises]}")
        if not self.is_consistent(premises):
            logging.info("Premises are inconsistent, skipping")
            return None

        premise_ids = [p["id"] for p in premises]
        premise_str = canonical_premise_str(premise_ids)

        # Check if we already have a valid argument
        if self._has_existing_valid_argument(premise_str):
            logging.info("Already have a valid argument, skipping")
            return None

        candidates = self.get_candidate_conclusions(premises, limit=limit)
        logging.info(f"Found {len(candidates)} candidate conclusions")

        for i, candidate in enumerate(candidates):
            start = time.time()
            if self.is_keeper_argument(premises, candidate):
                elapsed = int((time.time() - start) * 1000)
                difficulty = self._compute_difficulty_score(
                    premises=premises, conclusion=candidate, search_depth=i, eval_time_ms=elapsed
                )
                self.save_argument(premise_ids, candidate["id"], valid=True, difficulty=difficulty)
                logging.info(f"Saved valid argument {candidate['id']} with difficulty {difficulty}")

                # Find matching invalid arguments
                self.find_invalid_arguments(premises, candidate, candidates)
                return  # Exit after finding valid and its matching invalid arguments

        logging.info("No valid arguments found")
        return None

    def _has_existing_valid_argument(self, premise_str):
        """Check if we already have a valid argument for these premises."""
        return self.session.query(Argument).filter_by(premise_ids=premise_str, valid=True).count() > 0

    def save_argument(self, premise_ids, conclusion_id, valid, difficulty=None, source="z3"):
        """Save an argument to the database."""
        try:
            # Generate IDs
            premise_str = canonical_premise_str(premise_ids)
            arg_id = generate_argument_id(premise_ids, conclusion_id)

            # Create argument object
            argument = Argument(
                id=arg_id,
                premise_ids=premise_str,
                conclusion_id=conclusion_id,
                valid=valid,
                difficulty=difficulty,
                source=source,
                language=self.language,
            )

            # Save to database
            self.session.add(argument)
            self.session.commit()
            logging.info(f"Successfully saved {'valid' if valid else 'invalid'} argument: {arg_id}")
            return True

        except IntegrityError:
            self.session.rollback()
            logging.info(f"Argument already exists: {arg_id}")
            return False

        except Exception as e:
            self.session.rollback()
            logging.error(f"Error saving argument {arg_id}: {str(e)}")
            return False

    def _compute_difficulty_score(
        self, premises, conclusion, search_depth, eval_time_ms, w_len=0.2, w_d=0.5, w_t=0.002
    ):
        """Compute a difficulty score for the argument."""

        # Compute length score from premises and conclusion
        length_score = sum(len(p["form"]) for p in premises) + len(conclusion["form"])

        # Compute weighted sum
        score = (
            w_len * length_score  # Longer formulas are harder
            + w_d * search_depth  # More candidates tried means harder to find
            + w_t * eval_time_ms  # Longer validation time means harder to verify
        )

        return round(score, 2)

    def validate_and_save_argument(self, premises: List[Sentence], conclusion: Sentence, source: str = "manual"):
        """Validate and save a manually created argument."""

        try:
            logging.info(f"Validating argument with premises: {[p.id for p in premises]}")
            logging.info(f"Conclusion: {conclusion.id}")

            # Convert to dictionaries for validation
            premises_dict = [p.to_dict() for p in premises]
            conclusion_dict = conclusion.to_dict()

            # Check consistency
            if not self.is_consistent(premises_dict):
                logging.info("Premises are inconsistent")
                return None

            # Check if it's a keeper argument
            if not self.is_keeper_argument(premises_dict, conclusion_dict):
                logging.info("Argument is invalid")
                return None

            # Save the argument
            premise_ids = [p.id for p in premises]
            if not self.save_argument(premise_ids, conclusion.id, valid=True, difficulty=0, source=source):
                logging.error("Failed to save argument")
                return None

            # Return the saved argument
            premise_str = canonical_premise_str(premise_ids)
            saved_arg = (
                self.session.query(Argument).filter_by(premise_ids=premise_str, conclusion_id=conclusion.id).first()
            )

            if saved_arg:
                logging.info(f"Successfully validated and saved argument: {saved_arg.id}")
            else:
                logging.error("Argument was saved but not found in database")

            return saved_arg

        except Exception as e:
            logging.error(f"Error validating and saving argument: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def check_manual_argument(self, premise_ids: List[str], conclusion_id: str, source: str = "manual"):
        """Validate and save an argument given premise IDs and a conclusion ID."""

        session = db.session
        try:
            # Get the sentences
            premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
            if len(premises) != self.num_premises:
                logging.error(f"Expected {self.num_premises} premises, got {len(premises)}")
                return None

            conclusion = session.query(Sentence).filter_by(id=conclusion_id).first()
            if not conclusion:
                logging.error(f"Conclusion {conclusion_id} not found")
                return None

            # Convert to dictionaries for consistency
            premises_dict = [p.to_dict() for p in premises]
            conclusion_dict = conclusion.to_dict()

            # Validate and save the main argument
            if not self.is_consistent(premises_dict):
                logging.info("Premises are inconsistent")
                return None

            if not self.is_keeper_argument(premises_dict, conclusion_dict):
                logging.info("Argument is invalid")
                return None

            # Save the valid argument
            if not self.save_argument([p.id for p in premises], conclusion.id, valid=True, difficulty=0, source=source):
                logging.error("Failed to save valid argument")
                return None
            logging.info(f"Successfully saved valid argument with ID: {conclusion.id}")

            # Find and save matching invalid arguments
            candidates = self.get_candidate_conclusions(premises_dict, limit=1000)
            logging.info("Finding invalid conclusions...")
            self.find_invalid_arguments(premises_dict, conclusion_dict, candidates)

            return conclusion

        except Exception as e:
            logging.error(f"Error validating and saving argument: {e}")
            return None


def generate_arguments(target_valid_args=100, session=None, lexicon=None):
    """Generate arguments using a single process."""

    # Setup logging
    log_file = setup_logging("arg_generator")
    logging.info(f"Starting argument generation. Log file: {log_file}")
    logging.info(f"Parameters: target_valid_args={target_valid_args}, language={lexicon.language}")

    # Initialize generator with provided session or create new one
    if session is None:
        session = db.session
    evaluator = evaluate
    generator = ArgGenerator(session, evaluator, lexicon=lexicon)

    # Get initial count of valid arguments for this language
    initial_valid_count = session.query(Argument).filter_by(valid=True, language=lexicon.language).count()

    # Initialize counters
    stats = {
        "processed_premises": 0,
        "valid_count": 0,  # This will track only new valid arguments
        "invalid_count": 0,
        "start_time": datetime.now(),
        "batch_count": 0,
    }

    try:
        while stats["valid_count"] < target_valid_args:
            try:
                stats["batch_count"] += 1
                logging.info(f"\nStarting batch {stats['batch_count']}")

                for premises in generator.generate_premises(1000):
                    try:
                        # Process premises
                        generator.find_arguments(premises)

                        # Update stats - only count new valid arguments
                        stats["processed_premises"] += 1
                        current_valid_count = (
                            session.query(Argument).filter_by(valid=True, language=lexicon.language).count()
                        )
                        stats["valid_count"] = current_valid_count - initial_valid_count
                        stats["invalid_count"] = (
                            session.query(Argument).filter_by(valid=False, language=lexicon.language).count()
                        )

                        # Log progress
                        _log_progress(stats, target_valid_args)

                        if stats["valid_count"] >= target_valid_args:
                            break

                    except Exception as e:
                        logging.error(f"Error processing premises: {str(e)}")
                        session.rollback()
                        continue

            except Exception as e:
                logging.error(f"Error in batch processing: {str(e)}")
                session.rollback()
                continue

    except Exception as e:
        logging.error(f"Critical error in generate_arguments: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    finally:
        _log_final_summary(stats)


def _log_progress(stats, target_valid_args):
    """Log current progress of argument generation."""
    current_time = datetime.now()
    elapsed_time = (current_time - stats["start_time"]).total_seconds()

    logging.info(f"Processed premises: {stats['processed_premises']}")
    logging.info(f"Valid arguments found: {stats['valid_count']}/{target_valid_args}")
    logging.info(f"Invalid arguments found: {stats['invalid_count']}")
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

    if stats["processed_premises"] > 0:
        avg_time = elapsed_time / stats["processed_premises"]
        logging.info(f"Average time per premise: {avg_time:.2f} seconds")

    logging.info("=== End Update ===\n")


def _log_final_summary(stats):
    """Log final summary of argument generation."""
    total_time = (datetime.now() - stats["start_time"]).total_seconds()

    logging.info("\n=== Final Summary ===")
    logging.info(f"Total premises processed: {stats['processed_premises']}")
    logging.info(f"Total valid arguments: {stats['valid_count']}")
    logging.info(f"Total invalid arguments: {stats['invalid_count']}")
    logging.info(f"Total time: {total_time:.2f} seconds")

    if stats["processed_premises"] > 0:
        avg_time = total_time / stats["processed_premises"]
        logging.info(f"Average time per premise: {avg_time:.2f} seconds")

    logging.info("=== End Summary ===\n")


if __name__ == "__main__":
    # Import and create lexicon
    from Syntax.carroll_lexicon import CarrollLexicon

    lexicon = CarrollLexicon()

    # First check how many valid arguments we already have for this language
    existing_valid = db.session.query(Argument).filter_by(valid=True, language=lexicon.language).count()
    target_total = 3000  # Total number of valid arguments we want
    remaining = max(0, target_total - existing_valid)

    if remaining == 0:
        logging.info(f"Already have {existing_valid} valid arguments for {lexicon.language}, no need to generate more")
        exit(0)

    # Calculate how many each thread should generate
    num_threads = 6
    per_thread = math.ceil(remaining / num_threads)

    logging.info(f"Found {existing_valid} existing valid arguments for {lexicon.language}")
    logging.info(f"Need to generate {remaining} more valid arguments")
    logging.info(f"Each thread will target {per_thread} valid arguments")

    # Run parallel generators
    threads = []
    for i in range(num_threads):
        session = db.Session()
        thread = threading.Thread(
            target=lambda: generate_arguments(target_valid_args=per_thread, session=session, lexicon=lexicon),
            name=f"Generator-{i}-{lexicon.language}",
        )
        threads.append(thread)
        thread.start()
        logging.info(f"Started worker {i} for {lexicon.language}")

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
        logging.info(f"Worker {thread.name} completed")
