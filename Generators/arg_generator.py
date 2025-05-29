import random
from Database.DB import db
from Database.models import Sentence, Argument
from Utils.helpers import generate_argument_id, canonical_premise_str
from Semantics.eval import evaluate
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
import requests
from requests.exceptions import RequestException

# Constants and Configuration
CACHE_TTL = 3600
CACHE_MAXSIZE = 10000
BATCH_SIZE = 1000
DIFFICULTY_WEIGHTS = {"length": 0.2, "depth": 0.5, "time": 0.002}
MAX_RETRIES = 3
BATCH_RETRY_DELAY = 5
Z3_RETRY_DELAY = 2
Z3_MAX_RETRIES = 5
Z3_TIMEOUT = 30


class Z3ServerError(Exception):
    """Custom exception for Z3 server errors."""

    pass


def check_z3_server_health(host="localhost", port=8001, timeout=5):
    """Check if Z3 server is healthy and responding."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except RequestException:
        return False


def wait_for_z3_server(host="localhost", port=8001, max_retries=Z3_MAX_RETRIES, retry_delay=Z3_RETRY_DELAY):
    """Wait for Z3 server to become available."""
    for attempt in range(max_retries):
        if check_z3_server_health(host, port):
            logging.info("Z3 server is healthy")
            return True
        logging.warning(f"Z3 server not available, attempt {attempt + 1}/{max_retries}")
        time.sleep(retry_delay)
    return False


class ArgGenerator:
    """Generates logical arguments from a database of sentences."""

    def __init__(self, session, evaluator, lexicon1, lexicon2, num_premises=3, subset_percentage=0.2):
        """Initialize the argument generator."""
        self.session = session
        self.evaluate = evaluator
        self.num_premises = num_premises
        self.subset_percentage = subset_percentage
        self.lexicon1 = lexicon1
        self.lexicon2 = lexicon2
        self.language1 = lexicon1.language
        self.language2 = lexicon2.language
        self.z3_healthy = False

        # Initialize caches with optimized sizes
        self._premise_ast_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
        self._conclusion_from_subset_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
        self._consistency_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
        self._domain_constraint_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)  # Small cache for domain constraints
        self._valid_argument_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)  # Cache for valid argument checks

        # Initialize predicate groups
        self._initialize_predicate_groups()

        self._domain_constraint = None
        self._domain_constraint_ast = None

        # Check Z3 server health
        self._check_z3_health()

    # ===== Initialization and Setup Methods =====

    def _initialize_predicate_groups(self):
        """Initialize predicate groups for coherent story generation."""
        self.kind_predicates = {
            X for X in self.lexicon1.predicates.keys() if self.lexicon1.predicates[X]["semantic_type"] == "kind"
        }
        self.action_predicates = {
            X for X in self.lexicon1.predicates.keys() if self.lexicon1.predicates[X]["semantic_type"] == "action"
        }
        self.state_predicates = {
            X for X in self.lexicon1.predicates.keys() if self.lexicon1.predicates[X]["semantic_type"] == "state"
        }
        self.names = {X for X in self.lexicon1.names.keys()}

    def _get_domain_constraint(self):
        """Get and cache the domain constraint sentence and its AST."""
        try:
            cache_key = self.language1
            if cache_key in self._domain_constraint_cache:
                self._domain_constraint = self._domain_constraint_cache[cache_key]
                self._domain_constraint_ast = self._domain_constraint.ast
                logging.debug("Using cached domain constraint")
                return self._domain_constraint

            if self._domain_constraint is None:
                self._domain_constraint = (
                    self.session.query(Sentence).filter_by(type="domain_constraint", language=self.language1).first()
                )
                if self._domain_constraint:
                    self._domain_constraint_ast = self._domain_constraint.ast
                    # Cache the result
                    self._domain_constraint_cache[cache_key] = self._domain_constraint
                    logging.info(f"Loaded domain constraint with ID: {self._domain_constraint.id}")
                else:
                    logging.warning("No domain constraint found in database")
            return self._domain_constraint
        except Exception as e:
            logging.error(f"Error loading domain constraint: {e}")
            return None

    # ===== Cache Management =====

    def _get_premise_ast(self, premise_id):
        """Get and cache premise AST."""
        try:
            if premise_id not in self._premise_ast_cache:
                premise = self.session.query(Sentence).filter_by(id=premise_id).first()
                if premise:
                    self._premise_ast_cache[premise_id] = premise.ast
                else:
                    logging.warning(f"Premise {premise_id} not found")
            return self._premise_ast_cache[premise_id]
        except Exception as e:
            logging.error(f"Error getting AST for premise {premise_id}: {e}")
            return None

    # ===== Premise Generation =====

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
            # Get all available IDs for language 1
            available_ids = [
                id[0]
                for id in self.session.query(Sentence.id)
                .filter(Sentence.status == 0, Sentence.language == self.language1)
                .all()
            ]
            total_sentences = len(available_ids)
            logging.info(f"Found {total_sentences} candidate sentences for language {self.language1}")

            # Generate random IDs in batches
            for i in range(0, sample_size, BATCH_SIZE):
                current_batch = min(BATCH_SIZE, sample_size - i)
                if i % 1000 == 0:
                    logging.info(f"Processed {i} premise sets")

                # Generate a coherent predicate set for this batch
                selected_predicates, selected_names = self._generate_coherent_predicate_set()
                logging.debug(f"Generated coherent predicate set: {selected_predicates}")
                logging.debug(f"Selected names: {selected_names}")

                try:
                    # Get random IDs from actual available IDs
                    ids = random.sample(available_ids, current_batch * self.num_premises)

                    # Fetch sentences in batches
                    for j in range(0, len(ids), self.num_premises):
                        premise_ids = ids[j : j + self.num_premises]
                        premises = (
                            self.session.query(Sentence)
                            .filter(Sentence.id.in_(premise_ids), Sentence.language == self.language1)
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

    # ===== Argument Validation =====

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
            # Create cache key from sorted premise IDs
            premise_ids = tuple(sorted(p["id"] for p in premises))
            cache_key = premise_ids

            # Check cache first
            if cache_key in self._consistency_cache:
                result = self._consistency_cache[cache_key]
                logging.info(f"Using cached consistency result: {'consistent' if result else 'inconsistent'}")
                return result

            # Get domain constraint
            domain_constraint = (
                self.session.query(Sentence).filter_by(type="domain_constraint", language=self.language1).first()
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

            # Check satisfiability with safe evaluation
            try:
                result = self._safe_evaluate(joint_ast, convert_json=True) == "sat"
                # Cache the result
                self._consistency_cache[cache_key] = result
                logging.info(f"Premises are {'consistent' if result else 'inconsistent'}")
                return result
            except Z3ServerError as e:
                logging.error(f"Z3 server error during consistency check: {e}")
                return False
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

            # Get domain constraint
            domain_constraint = self._get_domain_constraint()
            if domain_constraint:
                logging.debug("Including domain constraint in subset check")

            # Build joint AST
            try:
                joint_ast = subset_asts[0]
                for ast in subset_asts[1:]:
                    joint_ast = ["and", joint_ast, ast]

                # Add domain constraint if it exists
                if domain_constraint:
                    joint_ast = ["and", joint_ast, self._domain_constraint_ast]
            except IndexError as e:
                logging.error(f"Error building joint AST: {e}")
                return False

            # Check if conclusion follows
            try:
                implication = ["and", joint_ast, ["not", conclusion_ast]]
                result = self._safe_evaluate(implication, convert_json=True) == "unsat"
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
                result = self._safe_evaluate(domain_implication, convert_json=True) == "unsat"
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
        """Check if the argument is a keeper (valid and not too trivial)."""
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
            # Create cache key from sorted premise IDs and conclusion AST
            premise_ids = tuple(sorted(p["id"] for p in premises))
            cache_key = (premise_ids, str(conclusion_ast))

            # Check cache first
            if cache_key in self._valid_argument_cache:
                result = self._valid_argument_cache[cache_key]
                logging.info(f"Using cached valid argument result: {'valid' if result else 'invalid'}")
                return result

            premise_asts = [self._get_premise_ast(p["id"]) for p in premises]
            if not all(premise_asts):
                logging.error("Failed to get all premise ASTs")
                return None

            # Get domain constraint
            domain_constraint = self._get_domain_constraint()
            if domain_constraint:
                logging.debug("Including domain constraint in validity check")

            # Build joint AST with premises
            joint_ast = premise_asts[0]
            for ast in premise_asts[1:]:
                joint_ast = ["and", joint_ast, ast]

            # Add domain constraint if it exists
            if domain_constraint:
                joint_ast = ["and", joint_ast, self._domain_constraint_ast]

            # Check if conclusion follows
            implication = ["and", joint_ast, ["not", conclusion_ast]]
            result = self._safe_evaluate(implication, convert_json=True) == "unsat"

            # Cache the result
            self._valid_argument_cache[cache_key] = result
            if result:
                logging.info(f"Found keeper argument requiring all premises: {[p['id'] for p in premises]}")
            else:
                logging.info("Argument is invalid - conclusion doesn't follow from all premises")
            return result

        except Exception as e:
            logging.error(f"Error in full validation: {e}")
            return None

    # ===== Argument Generation and Management Methods =====

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
            Sentence.language == self.language1,
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
        query = self.session.query(Sentence).filter(Sentence.status == 0, Sentence.language == self.language1)
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

            # First check if it follows from domain constraints alone
            if self._check_domain_constraints(candidate):
                logging.debug(f"Skipping candidate {candidate['id']} - follows from domain constraints alone")
                continue

            # Get conclusion AST
            conclusion_ast = self._get_premise_ast(candidate["id"])
            if not conclusion_ast:
                logging.error("Failed to get conclusion AST")
                continue

            # Check if it's actually invalid using _is_valid_argument directly
            is_valid = self._is_valid_argument(premises, conclusion_ast)
            if is_valid is None:
                logging.debug(f"Skipping candidate {candidate['id']} - validation failed")
                continue

            if not is_valid:
                # save invalid arguments for language 1
                self.save_argument(premise_ids, candidate["id"], valid=False, language=self.language1)
                logging.info(f"Saved invalid argument for language {self.language1}")

                # save invalid counterpart arguments for language 2
                counterpart_premise_ids = [p["counterpart_id"] for p in premises]
                counterpart_conclusion_id = candidate["counterpart_id"]
                self.save_argument(
                    counterpart_premise_ids, counterpart_conclusion_id, valid=False, language=self.language2
                )
                logging.info(f"Saved invalid counterpart argument for language {self.language2}")

                invalid_count += 1

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
                # save valid argument for language 1
                self.save_argument(
                    premise_ids, candidate["id"], valid=True, difficulty=difficulty, language=self.language1
                )
                logging.info(f"Saved valid argument for language {self.language1}")

                # save counterpart valid argument for language 2
                counterpart_premise_ids = [p["counterpart_id"] for p in premises]
                counterpart_conclusion_id = candidate["counterpart_id"]
                self.save_argument(
                    counterpart_premise_ids,
                    counterpart_conclusion_id,
                    valid=True,
                    difficulty=difficulty,
                    language=self.language2,
                )
                logging.info(f"Saved valid counterpart argument for language {self.language2}")

                # Find matching invalid arguments
                self.find_invalid_arguments(premises, candidate, candidates)
                return  # Exit after finding valid and its matching invalid arguments

        logging.info("No valid arguments found")
        return None

    # ===== Database and Storage Methods =====

    def _has_existing_valid_argument(self, premise_str):
        """Check if we already have a valid argument for these premises."""
        return (
            self.session.query(Argument).filter_by(premise_ids=premise_str, valid=True, language=self.language1).first()
            is not None
        )

    def save_argument(self, premise_ids, conclusion_id, valid, difficulty=None, source="z3", language=None):
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
                language=language,
            )

            # Save to database
            self.session.add(argument)
            self.session.commit()
            logging.info(f"Successfully saved {'valid' if valid else 'invalid'} argument: {arg_id}")
            return True

        except Exception as e:
            self.session.rollback()
            logging.error(f"Error saving argument {arg_id}: {str(e)}")
            return False

    def _compute_difficulty_score(
        self,
        premises,
        conclusion,
        search_depth,
        eval_time_ms,
        w_len=DIFFICULTY_WEIGHTS["length"],
        w_d=DIFFICULTY_WEIGHTS["depth"],
        w_t=DIFFICULTY_WEIGHTS["time"],
    ):
        """Compute a difficulty score for the argument."""
        # Compute length score from premises and conclusion
        length_score = sum(len(p["form"]) for p in premises) + len(conclusion["form"])
        num_premises = len(premises)

        # Compute weighted sum
        score = (
            w_len * length_score  # Longer formulas are harder
            + w_d * search_depth  # More candidates tried means harder to find
            + w_t * eval_time_ms  # Longer validation time means harder to verify
        ) * num_premises

        return round(score, 2)

    def check_manual_argument(self, premise_ids: List[str], conclusion_id: str, source: str = "manual"):
        """Validate and save an argument given premise IDs and a conclusion ID."""
        try:
            # Get the sentences
            premises = self.session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
            if len(premises) != self.num_premises:
                logging.error(f"Expected {self.num_premises} premises, got {len(premises)}")
                return None

            conclusion = self.session.query(Sentence).filter_by(id=conclusion_id).first()
            if not conclusion:
                logging.error(f"Conclusion {conclusion_id} not found")
                return None

            # Convert to dictionaries
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
            premise_ids = [p.id for p in premises]
            self.save_argument(
                premise_ids, conclusion.id, valid=True, difficulty=0, source=source, language=self.language1
            )
            logging.info("Successfully saved valid argument")

            counterpart_premise_ids = [p.counterpart_id for p in premises]
            counterpart_conclusion_id = conclusion.counterpart_id
            self.save_argument(
                counterpart_premise_ids,
                counterpart_conclusion_id,
                valid=True,
                difficulty=0,
                source=source,
                language=self.language2,
            )
            logging.info("Successfully saved valid counterpart argument")

            # Find and save matching invalid arguments
            candidates = self.get_candidate_conclusions(premises_dict, limit=1000)
            logging.info("Finding invalid conclusions...")
            self.find_invalid_arguments(premises_dict, conclusion_dict, candidates)

            return conclusion

        except Exception as e:
            logging.error(f"Error validating and saving argument: {e}")
            return None

    def _check_z3_health(self):
        """Check Z3 server health and attempt to reconnect if needed."""
        if not self.z3_healthy:
            self.z3_healthy = wait_for_z3_server()
            if not self.z3_healthy:
                logging.error("Z3 server is not available. Some operations may be limited.")
            else:
                logging.info("Z3 server is available and healthy")

    def _safe_evaluate(self, ast, convert_json=True):
        """Safely evaluate an AST with Z3 server, with retries and health checks."""
        if not self.z3_healthy:
            self._check_z3_health()
            if not self.z3_healthy:
                raise Z3ServerError("Z3 server is not available")

        for attempt in range(Z3_MAX_RETRIES):
            try:
                result = self.evaluate(ast, convert_json=convert_json)
                return result
            except RequestException as e:
                logging.warning(f"Z3 evaluation failed (attempt {attempt + 1}/{Z3_MAX_RETRIES}): {str(e)}")
                if attempt < Z3_MAX_RETRIES - 1:
                    time.sleep(Z3_RETRY_DELAY)
                    self._check_z3_health()
                else:
                    raise Z3ServerError(f"Failed to evaluate AST after {Z3_MAX_RETRIES} attempts")


# ===== Main Generation Functions =====


def generate_arguments(target_valid_args=100, session=None, lexicon1=None, lexicon2=None):
    """Generate arguments using a single process."""
    # setup logging
    log_file = setup_logging("arg_generator")
    logging.info(f"Starting argument generation. Log file: {log_file}")
    logging.info(
        f"Parameters: target_valid_args={target_valid_args}, languages={lexicon1.language} and {lexicon2.language}"
    )

    # initialize generator with provided session or create new one
    if session is None:
        session = db.Session()
    evaluator = evaluate
    generator = ArgGenerator(session, evaluator, lexicon1=lexicon1, lexicon2=lexicon2)

    # Check Z3 server health before starting
    if not generator.z3_healthy:
        logging.error("Cannot start argument generation - Z3 server is not available")
        return

    # get initial count of valid arguments for language 1 (since every argument has a counterpart)
    initial_valid_count = (
        session.query(Argument).filter(Argument.valid.is_(True), Argument.language == lexicon1.language).count()
    )

    # initialize counters
    stats = {
        "processed_premises": 0,
        "valid_count": 0,  # this will track only new valid arguments
        "invalid_count": 0,
        "start_time": datetime.now(),
        "batch_count": 0,
        "retry_count": 0,
        "z3_errors": 0,
    }

    try:
        while stats["valid_count"] < target_valid_args:
            try:
                stats["batch_count"] += 1
                logging.info(f"\nStarting batch {stats['batch_count']}")

                # Check Z3 health before processing batch
                if not generator.z3_healthy:
                    logging.warning("Z3 server became unavailable, attempting to reconnect...")
                    generator._check_z3_health()
                    if not generator.z3_healthy:
                        logging.error("Z3 server is still unavailable, waiting before retry...")
                        time.sleep(BATCH_RETRY_DELAY)
                        continue

                for premises in generator.generate_premises(BATCH_SIZE):
                    try:
                        # process premises
                        generator.find_arguments(premises)

                        # update stats - only count new valid arguments
                        stats["processed_premises"] += 1
                        current_valid_count = (
                            session.query(Argument)
                            .filter(Argument.valid.is_(True), Argument.language == lexicon1.language)
                            .count()
                        )
                        stats["valid_count"] = current_valid_count - initial_valid_count
                        stats["invalid_count"] = (
                            session.query(Argument)
                            .filter(Argument.valid.is_(False), Argument.language == lexicon1.language)
                            .count()
                        )

                        # log progress
                        _log_progress(stats, target_valid_args)

                        if stats["valid_count"] >= target_valid_args:
                            break

                    except Z3ServerError as e:
                        stats["z3_errors"] += 1
                        logging.error(f"Z3 server error: {e}")
                        if stats["z3_errors"] >= Z3_MAX_RETRIES:
                            logging.error("Too many Z3 server errors, stopping generation")
                            return
                        time.sleep(Z3_RETRY_DELAY)
                        continue
                    except Exception as e:
                        logging.error(f"Error processing premises: {str(e)}")
                        if stats["retry_count"] < MAX_RETRIES:
                            stats["retry_count"] += 1
                            logging.info(f"Retrying batch {stats['batch_count']} (attempt {stats['retry_count']})")
                            time.sleep(BATCH_RETRY_DELAY)
                            continue
                        else:
                            logging.error("Max retries reached, skipping batch")
                            stats["retry_count"] = 0
                            continue

            except Exception as e:
                logging.error(f"Error in batch processing: {str(e)}")
                if stats["retry_count"] < MAX_RETRIES:
                    stats["retry_count"] += 1
                    logging.info(f"Retrying batch {stats['batch_count']} (attempt {stats['retry_count']})")
                    time.sleep(BATCH_RETRY_DELAY)
                    continue
                else:
                    logging.error("Max retries reached, skipping batch")
                    stats["retry_count"] = 0
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
    logging.info(f"Z3 server errors: {stats['z3_errors']}")
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
    logging.info(f"Total Z3 server errors: {stats['z3_errors']}")
    logging.info(f"Total time: {total_time:.2f} seconds")

    if stats["processed_premises"] > 0:
        avg_time = total_time / stats["processed_premises"]
        logging.info(f"Average time per premise: {avg_time:.2f} seconds")

    logging.info("=== End Summary ===\n")


if __name__ == "__main__":
    # import and create lexicons
    from Syntax.carroll_lexicon import CarrollLexicon
    from Syntax.english_lexicon import EnglishLexicon

    lexicon1 = CarrollLexicon()
    lexicon2 = EnglishLexicon()

    # first check how many valid arguments we already have for language 1
    existing_valid = db.get_valid_argument_count(lexicon1.language)
    target_total = 3000  # total number of valid arguments we want
    remaining = max(0, target_total - existing_valid)

    if remaining == 0:
        logging.info(
            f"Already have {existing_valid} valid arguments for {lexicon1.language} (and their counterparts in {lexicon2.language}), no need to generate more"
        )
        exit(0)

    # calculate how many each thread should generate
    num_threads = 6
    per_thread = math.ceil(remaining / num_threads)

    logging.info(f"Found {existing_valid} existing valid arguments for {lexicon1.language}")
    logging.info(f"Need to generate {remaining} more valid arguments")
    logging.info(f"Each thread will target {per_thread} valid arguments")

    # Run parallel generators
    threads = []
    for i in range(num_threads):
        # Create a new session for this thread
        session = db.Session()
        thread = threading.Thread(
            target=lambda s=session: generate_arguments(
                target_valid_args=per_thread, session=s, lexicon1=lexicon1, lexicon2=lexicon2
            ),
            name=f"Generator-{i}-{lexicon1.language}-{lexicon2.language}",
        )
        threads.append(thread)
        thread.start()
        logging.info(f"Started worker {i} for {lexicon1.language} and {lexicon2.language}")

    # wait for all threads to complete
    for thread in threads:
        thread.join()
        logging.info(f"Worker {thread.name} completed")
