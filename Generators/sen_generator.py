import itertools
import time
import random
from Database.DB import db
from Syntax.parse import parser
from Syntax.transform import transformer
from Semantics.eval import evaluate
from Utils.normalize import normalize_logical_form
from Utils.logging_config import setup_logging
import logging

# Set up logging for this module
log_file = setup_logging("sentence_generator")
logger = logging.getLogger(__name__)


class SentenceGenerator:
    """A class for generating sentences in different languages and logical forms."""

    def __init__(self, lexicon1, lexicon2, db=db):
        """Initialize the sentence generator with two lexicons."""
        self.lexicon1 = lexicon1
        self.lexicon2 = lexicon2
        self.counter = 0
        self.db = db
        self.language1 = lexicon1.language if lexicon1.language is not None else None
        self.language2 = lexicon2.language if lexicon2.language is not None else None

    # ===== Private Helper Methods =====

    def _sentence_exists(self, form, type=None, subtype=None, language=None):
        """Check if a sentence with the given form already exists in the database."""
        try:
            normalized_form = normalize_logical_form(form)
            return self.db.sentence_exists(normalized_form, type, subtype, language=language)
        except Exception as e:
            logger.error(f"Error checking for existing sentence: {e}")
            return True  # if error return true to avoid duplicate sentences

    def _parse_ast(self, form):
        """Parse a logical form into an AST."""
        logger.info(f"Parsing AST for: {form}")
        try:
            tree = parser.parse(form)
            ast = transformer.transform(tree)
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            ast = None
        return ast

    def _get_status(self, raw_ast):
        """Get the logical status of an AST."""
        result = evaluate(raw_ast)
        if result == "unsat":
            return -1  # logical falsehood
        elif evaluate(("not", raw_ast)) == "unsat":
            return 1  # theorem
        elif result == "sat":
            return 0  # contingent
        else:
            return None  # evaluation failed or unknown

    def _ast_to_json_compatible(self, raw_ast):
        """Convert AST to JSON-compatible format."""
        if isinstance(raw_ast, tuple):
            return [self._ast_to_json_compatible(x) for x in raw_ast]
        return raw_ast

    def _lowercase_except_names(self, sentence, lexicon):
        """Convert sentence to lowercase except for proper names."""
        names = {data["name"] for data in lexicon.names.values()}
        words = sentence.split()
        if not words:
            return sentence
        return " ".join(word if word in names else word.lower() for word in words)

    def _capitalize(self, sentence):
        """Capitalize the first letter of a sentence."""
        return sentence[0].upper() + sentence[1:] if sentence else sentence

    # ===== Database Operations =====

    def add_entry(self, sentence, type, subtype, soa, form, base=False, counterpart_id=None, language=None):
        """Add a new sentence entry to the database."""
        timestamp = int(time.time())
        # normalize the form to escaped Unicode format
        raw_ast = self._parse_ast(form)
        normalized_form = normalize_logical_form(form)

        # check if the sentence already exists
        if self._sentence_exists(normalized_form, type, subtype, language=language):
            logger.info(f"Sentence already exists: {sentence}")
            return

        status = self._get_status(raw_ast)
        try:
            self.db.add_sentence(
                sentence=sentence,
                type=type,
                subtype=subtype,
                soa=soa,
                form=normalized_form,
                ast=self._ast_to_json_compatible(raw_ast),
                base=1 if base else 0,
                status=status,
                language=language,
                counterpart_id=counterpart_id,
                time_created=timestamp,
            )
            self.counter += 1
            logger.info(f"{self.counter}: {sentence}")

        except Exception as e:
            logger.error(f"Error adding entry: {e}")

    def get_last_inserted_id(self):
        """Get the ID of the last inserted sentence."""
        try:
            return self.db.get_last_inserted_id()
        except Exception as e:
            logger.error(f"Error getting last inserted ID: {e}")
            return None

    def update_sentence_counterpart(self, sentence_id, counterpart_id):
        """Update a sentence's counterpart_id."""
        try:
            self.db.update_sentence_counterpart(sentence_id, counterpart_id)
            logger.info(f"Updated sentence {sentence_id} with counterpart {counterpart_id}")
        except Exception as e:
            logger.error(f"Error updating sentence counterpart: {e}")

    def get_entries(self):
        """Get all sentences from the database."""
        try:
            return self.db.get_all_sentences()
        except Exception as e:
            logger.error(f"Error retrieving entries: {e}")

    def get_base_entries(self, language=None):
        """Get base entries from the database."""
        try:
            return self.db.get_base_entries(language=language)
        except Exception as e:
            logger.error(f"Error retrieving base entries: {e}")

    # ===== Sentence Generation Methods =====

    def generate_domain_constraint(self):
        """Generate domain constraint sentences."""
        kind_predicates1 = {
            k: v for k, v in self.lexicon1.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        kind_predicates2 = {
            k: v for k, v in self.lexicon2.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }

        # build the English sentence
        or_clause1 = ""
        soa1 = {}
        for kind_symbol, kind_data in kind_predicates1.items():
            soa1[kind_symbol] = kind_data["template"]
            kind_noun = kind_data["template"].replace("[1] is a ", "").strip()
            art = "an" if kind_noun[0].lower() in "aeiou" else "a"
            or_clause1 += f"{art} {kind_noun}, or "

        or_clause1 = or_clause1.rstrip(", or ")
        sentence1 = f"Everything is {or_clause1} (exclusively), and there's at least one of each."

        # build the Carroll sentence
        or_clause2 = ""
        soa2 = {}
        for kind_symbol, kind_data in kind_predicates2.items():
            soa2[kind_symbol] = kind_data["template"]
            kind_noun = kind_data["template"].replace("[1] is a ", "").strip()
            art = "an" if kind_noun[0].lower() in "aeiou" else "a"
            or_clause2 += f"{art} {kind_noun}, or "

        or_clause2 = or_clause2.rstrip(", or ")
        sentence2 = f"Everything is {or_clause2} (exclusively), and there's at least one of each."

        kinds = list(kind_predicates1.keys())  # use same order for both lexicons

        def construct_universal_clause(kinds):
            # start with the last kind
            result = f"{kinds[-1]}x"

            # add the rest in reverse order with proper nesting
            for kind in reversed(kinds[:-1]):
                result = f"({kind}x∨{result})"

            return f"∀x{result}"

        def construct_exclusivity_clauses(kinds):
            clauses = []
            for i in range(len(kinds)):
                for j in range(i + 1, len(kinds)):
                    clause = f"¬∃x({kinds[i]}x∧{kinds[j]}x)"
                    clauses.append(clause)
            # start with the last clause
            result = clauses[-1]
            # add the rest in reverse order with proper nesting
            for clause in reversed(clauses[:-1]):
                result = f"({clause} ∧ {result})"

            return result

        def construct_existence_clauses(kinds):
            # start with the last kind
            result = f"∃x{kinds[-1]}x"

            # add the rest in reverse order with proper nesting
            for kind in reversed(kinds[:-1]):
                result = f"(∃x{kind}x ∧ {result})"

            return result

        conjunct1 = construct_universal_clause(kinds)
        conjunct2 = construct_exclusivity_clauses(kinds)
        conjunct3 = construct_existence_clauses(kinds)
        form = f"(({conjunct1} ∧ {conjunct2}) ∧ {conjunct3})"

        # add English domain constraint
        entry1 = {
            "sentence": sentence1,
            "type": "domain_constraint",
            "subtype": None,
            "soa": soa1,
            "form": form,
            "language": self.language1,
        }
        self.add_entry(**entry1)
        sentence1_id = self.db.get_last_inserted_id()

        # add Carroll domain constraint
        entry2 = {
            "sentence": sentence2,
            "type": "domain_constraint",
            "subtype": None,
            "soa": soa2,
            "form": form,
            "counterpart_id": sentence1_id,
            "language": self.language2,
        }
        self.add_entry(**entry2)
        sentence2_id = self.db.get_last_inserted_id()
        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

    def generate_atomic_sentences(self, vp_ellipsis=True):
        """Generate atomic sentences."""
        # generate sentences using lexicon1
        for pred_symbol, pred in self.lexicon1.predicates.items():
            arity = pred["arity"]
            template = pred["template"]
            structure = pred["structure"]
            semantic_type = pred["semantic_type"]
            neg_template = pred["negated_template"]
            names = self.lexicon1.names

            # get corresponding predicate from lexicon2
            pred2 = self.lexicon2.predicates.get(pred_symbol)
            if not pred2:
                continue  # skip if no counterpart predicate exists

            if arity == 1:
                for name_symbol in self.lexicon1.names:
                    name1 = self.lexicon1.get_name(name_symbol)
                    name2 = self.lexicon2.get_name(name_symbol)

                    form = f"{pred_symbol}{name_symbol}"
                    soa1 = {pred_symbol: template, name_symbol: name1}
                    soa2 = {pred_symbol: pred2["template"], name_symbol: name2}

                    # generate sentence in lexicon1
                    sentence1 = template.replace("[1]", name1) + "."
                    # generate counterpart in lexicon2
                    sentence2 = pred2["template"].replace("[1]", name2) + "."

                    # add first sentence and get its ID
                    self.add_entry(
                        sentence=sentence1,
                        type="atomic",
                        subtype="monadic",
                        soa=soa1,
                        form=form,
                        base=True,
                        language=self.language1,
                    )
                    # get the ID of the first sentence
                    sentence1_id = self.db.get_last_inserted_id()

                    # add second sentence with first sentence's ID as its counterpart
                    self.add_entry(
                        sentence=sentence2,
                        type="atomic",
                        subtype="monadic",
                        soa=soa2,
                        form=form,
                        base=True,
                        counterpart_id=sentence1_id,
                        language=self.language2,
                    )
                    # get the ID of the second sentence
                    sentence2_id = self.db.get_last_inserted_id()

                    # update first sentence with second sentence's ID as its counterpart
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # vp ellipsis cases
                    if vp_ellipsis:
                        for name_symbol2 in names:
                            if semantic_type == "kind":
                                continue
                            if name_symbol != name_symbol2:
                                name1_2 = self.lexicon1.get_name(name_symbol2)
                                name2_2 = self.lexicon2.get_name(name_symbol2)

                                expanded_soa1 = {
                                    pred_symbol: template,
                                    name_symbol: name1,
                                    name_symbol2: name1_2,
                                }
                                expanded_soa2 = {
                                    pred_symbol: pred2["template"],
                                    name_symbol: name2,
                                    name_symbol2: name2_2,
                                }

                                # generate disjunction sentences
                                if structure.startswith("copula"):
                                    disjunction_vpe1 = template.replace("[1] is", f"{name1} or {name1_2} are") + "."
                                    disjunction_vpe2 = (
                                        pred2["template"].replace("[1] is", f"{name2} or {name2_2} are") + "."
                                    )
                                else:
                                    disjunction_vpe1 = template.replace("[1]", f"{name1} or {name1_2}") + "."
                                    disjunction_vpe2 = pred2["template"].replace("[1]", f"{name2} or {name2_2}") + "."

                                disjunction_form = f"({pred_symbol}{name_symbol}∨{pred_symbol}{name_symbol2})"

                                # add first disjunction and get its ID
                                self.add_entry(
                                    disjunction_vpe1,
                                    "disjunction",
                                    "vp_ellipsis",
                                    expanded_soa1,
                                    disjunction_form,
                                    language=self.language1,
                                )
                                disjunction1_id = self.db.get_last_inserted_id()

                                # add second disjunction with first disjunction's ID as its counterpart
                                self.add_entry(
                                    disjunction_vpe2,
                                    "disjunction",
                                    "vp_ellipsis",
                                    expanded_soa2,
                                    disjunction_form,
                                    counterpart_id=disjunction1_id,
                                    language=self.language2,
                                )
                                disjunction2_id = self.db.get_last_inserted_id()

                                # update first disjunction with second disjunction's ID as its counterpart
                                self.db.update_sentence_counterpart(disjunction1_id, disjunction2_id)

                                # generate conjunction sentences
                                if structure.startswith("copula"):
                                    conjunction_vpe1 = template.replace("[1] is", f"{name1} and {name1_2} are") + "."
                                    conjunction_vpe2 = (
                                        pred2["template"].replace("[1] is", f"{name2} and {name2_2} are") + "."
                                    )
                                else:
                                    conjunction_vpe1 = template.replace("[1]", f"{name1} and {name1_2}") + "."
                                    conjunction_vpe2 = pred2["template"].replace("[1]", f"{name2} and {name2_2}") + "."

                                conjunction_form = f"({pred_symbol}{name_symbol}∧{pred_symbol}{name_symbol2})"

                                # add first conjunction and get its ID
                                self.add_entry(
                                    conjunction_vpe1,
                                    "conjunction",
                                    "vp_ellipsis",
                                    expanded_soa1,
                                    conjunction_form,
                                    language=self.language1,
                                )
                                conjunction1_id = self.db.get_last_inserted_id()

                                # add second conjunction with first conjunction's ID as its counterpart
                                self.add_entry(
                                    conjunction_vpe2,
                                    "conjunction",
                                    "vp_ellipsis",
                                    expanded_soa2,
                                    conjunction_form,
                                    counterpart_id=conjunction1_id,
                                    language=self.language2,
                                )
                                conjunction2_id = self.db.get_last_inserted_id()

                                # update first conjunction with second conjunction's ID as its counterpart
                                self.db.update_sentence_counterpart(conjunction1_id, conjunction2_id)

                    # generate negated sentences
                    neg_form = f"\u00ac{pred_symbol}{name_symbol}"
                    neg_sentence1 = neg_template.replace("[1]", name1) + "."
                    neg_sentence2 = pred2["negated_template"].replace("[1]", name2) + "."

                    # add first negated sentence and get its ID
                    self.add_entry(
                        neg_sentence1,
                        "negation",
                        "monadic",
                        soa1,
                        neg_form,
                        base=True,
                        language=self.language1,
                    )
                    neg1_id = self.db.get_last_inserted_id()

                    # add second negated sentence with first negated sentence's ID as its counterpart
                    self.add_entry(
                        neg_sentence2,
                        "negation",
                        "monadic",
                        soa2,
                        neg_form,
                        base=True,
                        counterpart_id=neg1_id,
                        language=self.language2,
                    )
                    neg2_id = self.db.get_last_inserted_id()

                    # update first negated sentence with second negated sentence's ID as its counterpart
                    self.db.update_sentence_counterpart(neg1_id, neg2_id)

            elif arity == 2:
                for name1 in self.lexicon1.names:
                    for name2 in self.lexicon1.names:
                        name1_1 = self.lexicon1.get_name(name1)
                        name1_2 = self.lexicon1.get_name(name2)
                        name2_1 = self.lexicon2.get_name(name1)
                        name2_2 = self.lexicon2.get_name(name2)

                        form = f"{pred_symbol}{name1}{name2}"
                        soa1 = {pred_symbol: template, name1: name1_1, name2: name1_2}
                        soa2 = {pred_symbol: pred2["template"], name1: name2_1, name2: name2_2}

                        # generate atomic sentences
                        sentence1 = template.replace("[1]", name1_1).replace("[2]", name1_2) + "."
                        sentence2 = pred2["template"].replace("[1]", name2_1).replace("[2]", name2_2) + "."

                        # add first atomic sentence and get its ID
                        self.add_entry(
                            sentence1,
                            "atomic",
                            "dyadic",
                            soa1,
                            form,
                            base=True,
                            language=self.language1,
                        )
                        atomic1_id = self.db.get_last_inserted_id()

                        # add second atomic sentence with first atomic sentence's ID as its counterpart
                        self.add_entry(
                            sentence2,
                            "atomic",
                            "dyadic",
                            soa2,
                            form,
                            base=True,
                            counterpart_id=atomic1_id,
                            language=self.language2,
                        )
                        atomic2_id = self.db.get_last_inserted_id()

                        # update first atomic sentence with second atomic sentence's ID as its counterpart
                        self.db.update_sentence_counterpart(atomic1_id, atomic2_id)

                        # generate negated sentences
                        neg_form = f"\u00ac{pred_symbol}{name1}{name2}"
                        neg_sentence1 = neg_template.replace("[1]", name1_1).replace("[2]", name1_2) + "."
                        neg_sentence2 = pred2["negated_template"].replace("[1]", name2_1).replace("[2]", name2_2) + "."

                        # add first negated sentence and get its ID
                        self.add_entry(
                            neg_sentence1,
                            "negation",
                            "monadic",
                            soa1,
                            neg_form,
                            base=True,
                            language=self.language1,
                        )
                        neg1_id = self.db.get_last_inserted_id()

                        # add second negated sentence with first negated sentence's ID as its counterpart
                        self.add_entry(
                            neg_sentence2,
                            "negation",
                            "monadic",
                            soa2,
                            neg_form,
                            base=True,
                            counterpart_id=neg1_id,
                            language=self.language2,
                        )
                        neg2_id = self.db.get_last_inserted_id()

                        # update first negated sentence with second negated sentence's ID as its counterpart
                        self.db.update_sentence_counterpart(neg1_id, neg2_id)

    def generate_simple_quantified_sentences(self):
        """Generate simple quantified sentences."""
        kind_predicates1 = {
            k: v for k, v in self.lexicon1.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        kind_predicates2 = {
            k: v for k, v in self.lexicon2.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }

        for pred_symbol, pred_data in self.lexicon1.predicates.items():
            if pred_symbol in kind_predicates1:
                continue  # skip restrictors as main predicates

            # get corresponding predicate from lexicon2
            pred2 = self.lexicon2.predicates.get(pred_symbol)
            if not pred2:
                continue  # skip if no counterpart predicate exists

            arity = pred_data["arity"]
            pred_template1 = pred_data["template"]
            pred_template2 = pred2["template"]

            for restrictor_symbol, restrictor_data in kind_predicates1.items():
                # get corresponding restrictor from lexicon2
                restrictor2 = kind_predicates2.get(restrictor_symbol)
                if not restrictor2:
                    continue  # skip if no counterpart restrictor exists

                restrictor_label1 = restrictor_data["template"]
                restrictor_label2 = restrictor2["template"]
                kind_noun1 = restrictor_label1.replace("[1] is a ", "").strip()
                kind_noun2 = restrictor_label2.replace("[1] is a ", "").strip()
                art1 = "An" if kind_noun1[0].lower() in "aeiou" else "A"
                art2 = "An" if kind_noun2[0].lower() in "aeiou" else "A"

                if arity == 1:
                    # get the base template without the subject
                    base1 = pred_template1.replace("[1]", "")
                    base2 = pred_template2.replace("[1]", "")

                    predicate_singular1 = base1.strip()
                    predicate_singular2 = base2.strip()

                    predicate_plural1 = (
                        base1.replace(" is ", " are ")
                        if pred_data["structure"].startswith("copula")
                        else predicate_singular1
                    ).strip()

                    predicate_plural2 = (
                        base2.replace(" is ", " are ")
                        if pred2["structure"].startswith("copula")
                        else predicate_singular2
                    ).strip()

                    # universal affirmative - use singular form
                    universal_affirmative1 = {
                        "sentence": f"Every {kind_noun1} {predicate_singular1}.",
                        "type": "quantified",
                        "subtype": "universal_affirmative",
                        "soa": {
                            restrictor_symbol: restrictor_label1,
                            pred_symbol: pred_template1,
                        },
                        "form": f"∀x({restrictor_symbol}x→{pred_symbol}x)",
                        "base": True,
                        "language": self.language1,
                    }
                    self.add_entry(**universal_affirmative1)
                    sentence1_id = self.db.get_last_inserted_id()

                    universal_affirmative2 = {
                        "sentence": f"Every {kind_noun2} {predicate_singular2}.",
                        "type": "quantified",
                        "subtype": "universal_affirmative",
                        "soa": {
                            restrictor_symbol: restrictor_label2,
                            pred_symbol: pred_template2,
                        },
                        "form": f"∀x({restrictor_symbol}x→{pred_symbol}x)",
                        "base": True,
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**universal_affirmative2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # particular affirmative - use singular form
                    particular_affirmative1 = {
                        "sentence": f"{art1} {kind_noun1} {predicate_singular1}.",
                        "type": "quantified",
                        "subtype": "particular_affirmative",
                        "soa": {
                            restrictor_symbol: restrictor_label1,
                            pred_symbol: pred_template1,
                        },
                        "form": f"∃x({restrictor_symbol}x∧{pred_symbol}x)",
                        "base": True,
                        "language": self.language1,
                    }
                    self.add_entry(**particular_affirmative1)
                    sentence1_id = self.db.get_last_inserted_id()

                    particular_affirmative2 = {
                        "sentence": f"{art2} {kind_noun2} {predicate_singular2}.",
                        "type": "quantified",
                        "subtype": "particular_affirmative",
                        "soa": {
                            restrictor_symbol: restrictor_label2,
                            pred_symbol: pred_template2,
                        },
                        "form": f"∃x({restrictor_symbol}x∧{pred_symbol}x)",
                        "base": True,
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**particular_affirmative2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # universal negative - use plural form
                    universal_negative1 = {
                        "sentence": f"No {kind_noun1}s {predicate_plural1}.",
                        "type": "quantified",
                        "subtype": "universal_negative",
                        "soa": {
                            restrictor_symbol: restrictor_label1,
                            pred_symbol: pred_template1,
                        },
                        "form": f"¬∃x({restrictor_symbol}x∧{pred_symbol}x)",
                        "base": True,
                        "language": self.language1,
                    }
                    self.add_entry(**universal_negative1)
                    sentence1_id = self.db.get_last_inserted_id()

                    universal_negative2 = {
                        "sentence": f"No {kind_noun2}s {predicate_plural2}.",
                        "type": "quantified",
                        "subtype": "universal_negative",
                        "soa": {
                            restrictor_symbol: restrictor_label2,
                            pred_symbol: pred_template2,
                        },
                        "form": f"¬∃x({restrictor_symbol}x∧{pred_symbol}x)",
                        "base": True,
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**universal_negative2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # particular negative - use plural form
                    particular_negative1 = {
                        "sentence": f"Not all {kind_noun1}s {predicate_plural1}.",
                        "type": "quantified",
                        "subtype": "particular_negative",
                        "soa": {
                            restrictor_symbol: restrictor_label1,
                            pred_symbol: pred_template1,
                        },
                        "form": f"∃x({restrictor_symbol}x∧¬{pred_symbol}x)",
                        "base": True,
                        "language": self.language1,
                    }
                    self.add_entry(**particular_negative1)
                    sentence1_id = self.db.get_last_inserted_id()

                    particular_negative2 = {
                        "sentence": f"Not all {kind_noun2}s {predicate_plural2}.",
                        "type": "quantified",
                        "subtype": "particular_negative",
                        "soa": {
                            restrictor_symbol: restrictor_label2,
                            pred_symbol: pred_template2,
                        },
                        "form": f"∃x({restrictor_symbol}x∧¬{pred_symbol}x)",
                        "base": True,
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**particular_negative2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # only restrictor - use plural form
                    only_restrictor1 = {
                        "sentence": f"Only {kind_noun1}s {predicate_plural1}.",
                        "type": "quantified",
                        "subtype": "only_restrictor",
                        "soa": {
                            restrictor_symbol: restrictor_label1,
                            pred_symbol: pred_template1,
                        },
                        "form": f"∀x({pred_symbol}x→{restrictor_symbol}x)",
                        "base": True,
                        "language": self.language1,
                    }
                    self.add_entry(**only_restrictor1)
                    sentence1_id = self.db.get_last_inserted_id()

                    only_restrictor2 = {
                        "sentence": f"Only {kind_noun2}s {predicate_plural2}.",
                        "type": "quantified",
                        "subtype": "only_restrictor",
                        "soa": {
                            restrictor_symbol: restrictor_label2,
                            pred_symbol: pred_template2,
                        },
                        "form": f"∀x({pred_symbol}x→{restrictor_symbol}x)",
                        "base": True,
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**only_restrictor2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                elif arity == 2:
                    for name_sym, name_data in self.lexicon1.names.items():
                        name1 = name_data["name"]
                        name2 = self.lexicon2.get_name(name_sym)

                        base1 = pred_template1.replace("[1]", "").replace("[2]", name1)
                        base2 = pred_template2.replace("[1]", "").replace("[2]", name2)

                        predicate_singular1 = base1.strip()
                        predicate_singular2 = base2.strip()

                        predicate_plural1 = (
                            base1.replace(" is ", " are ")
                            if pred_data["structure"].startswith("copula")
                            else predicate_singular1
                        ).strip()

                        predicate_plural2 = (
                            base2.replace(" is ", " are ")
                            if pred2["structure"].startswith("copula")
                            else predicate_singular2
                        ).strip()

                        # universal affirmative
                        universal_affirmative1 = {
                            "sentence": f"Every {kind_noun1} {predicate_singular1}.",
                            "type": "quantified",
                            "subtype": "universal_affirmative",
                            "soa": {
                                restrictor_symbol: restrictor_label1,
                                pred_symbol: pred_template1,
                                name_sym: name1,
                            },
                            "form": f"∀x({restrictor_symbol}x→{pred_symbol}x{name_sym})",
                            "base": True,
                            "language": self.language1,
                        }
                        self.add_entry(**universal_affirmative1)
                        sentence1_id = self.db.get_last_inserted_id()

                        universal_affirmative2 = {
                            "sentence": f"Every {kind_noun2} {predicate_singular2}.",
                            "type": "quantified",
                            "subtype": "universal_affirmative",
                            "soa": {
                                restrictor_symbol: restrictor_label2,
                                pred_symbol: pred_template2,
                                name_sym: name2,
                            },
                            "form": f"∀x({restrictor_symbol}x→{pred_symbol}x{name_sym})",
                            "base": True,
                            "counterpart_id": sentence1_id,
                            "language": self.language2,
                        }
                        self.add_entry(**universal_affirmative2)
                        sentence2_id = self.db.get_last_inserted_id()
                        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                        # particular affirmative
                        particular_affirmative1 = {
                            "sentence": f"{art1} {kind_noun1} {predicate_singular1}.",
                            "type": "quantified",
                            "subtype": "particular_affirmative",
                            "soa": {
                                restrictor_symbol: restrictor_label1,
                                pred_symbol: pred_template1,
                                name_sym: name1,
                            },
                            "form": f"∃x({restrictor_symbol}x∧{pred_symbol}x{name_sym})",
                            "base": True,
                            "language": self.language1,
                        }
                        self.add_entry(**particular_affirmative1)
                        sentence1_id = self.db.get_last_inserted_id()

                        particular_affirmative2 = {
                            "sentence": f"{art2} {kind_noun2} {predicate_singular2}.",
                            "type": "quantified",
                            "subtype": "particular_affirmative",
                            "soa": {
                                restrictor_symbol: restrictor_label2,
                                pred_symbol: pred_template2,
                                name_sym: name2,
                            },
                            "form": f"∃x({restrictor_symbol}x∧{pred_symbol}x{name_sym})",
                            "base": True,
                            "counterpart_id": sentence1_id,
                            "language": self.language2,
                        }
                        self.add_entry(**particular_affirmative2)
                        sentence2_id = self.db.get_last_inserted_id()
                        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                        # universal negative
                        universal_negative1 = {
                            "sentence": f"No {kind_noun1}s {predicate_plural1}.",
                            "type": "quantified",
                            "subtype": "universal_negative",
                            "soa": {
                                restrictor_symbol: restrictor_label1,
                                pred_symbol: pred_template1,
                                name_sym: name1,
                            },
                            "form": f"¬∃x({restrictor_symbol}x∧{pred_symbol}x{name_sym})",
                            "base": True,
                            "language": self.language1,
                        }
                        self.add_entry(**universal_negative1)
                        sentence1_id = self.db.get_last_inserted_id()

                        universal_negative2 = {
                            "sentence": f"No {kind_noun2}s {predicate_plural2}.",
                            "type": "quantified",
                            "subtype": "universal_negative",
                            "soa": {
                                restrictor_symbol: restrictor_label2,
                                pred_symbol: pred_template2,
                                name_sym: name2,
                            },
                            "form": f"¬∃x({restrictor_symbol}x∧{pred_symbol}x{name_sym})",
                            "base": True,
                            "counterpart_id": sentence1_id,
                            "language": self.language2,
                        }
                        self.add_entry(**universal_negative2)
                        sentence2_id = self.db.get_last_inserted_id()
                        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                        # particular negative
                        particular_negative1 = {
                            "sentence": f"Not all {kind_noun1}s {predicate_plural1}.",
                            "type": "quantified",
                            "subtype": "particular_negative",
                            "soa": {
                                restrictor_symbol: restrictor_label1,
                                pred_symbol: pred_template1,
                                name_sym: name1,
                            },
                            "form": f"∃x({restrictor_symbol}x∧¬{pred_symbol}x{name_sym})",
                            "base": True,
                            "language": self.language1,
                        }
                        self.add_entry(**particular_negative1)
                        sentence1_id = self.db.get_last_inserted_id()

                        particular_negative2 = {
                            "sentence": f"Not all {kind_noun2}s {predicate_plural2}.",
                            "type": "quantified",
                            "subtype": "particular_negative",
                            "soa": {
                                restrictor_symbol: restrictor_label2,
                                pred_symbol: pred_template2,
                                name_sym: name2,
                            },
                            "form": f"∃x({restrictor_symbol}x∧¬{pred_symbol}x{name_sym})",
                            "base": True,
                            "counterpart_id": sentence1_id,
                            "language": self.language2,
                        }
                        self.add_entry(**particular_negative2)
                        sentence2_id = self.db.get_last_inserted_id()
                        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

    def generate_multiply_quantified_sentences(self):
        """Generate multiply quantified sentences."""
        # generate sentences with multiple quantifiers
        kind_predicates1 = {
            k: v for k, v in self.lexicon1.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        kind_predicates2 = {
            k: v for k, v in self.lexicon2.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        dyadic_predicates1 = {}
        dyadic_predicates2 = {}
        for pred_symbol, pred_data in self.lexicon1.predicates.items():
            if pred_data["arity"] == 2:
                dyadic_predicates1[pred_symbol] = pred_data
                # get corresponding predicate from lexicon2
                pred2 = self.lexicon2.predicates.get(pred_symbol)
                if pred2:
                    dyadic_predicates2[pred_symbol] = pred2

        for pred_symbol, pred_data in dyadic_predicates1.items():
            if pred_symbol in kind_predicates1:
                continue
            relation1 = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
            relation2 = dyadic_predicates2[pred_symbol]["template"].replace("[1]", "").replace("[2]", "").strip()

            for restrictor_symbol1, restrictor_data1 in kind_predicates1.items():
                # get corresponding restrictor from lexicon2
                restrictor2_1 = kind_predicates2.get(restrictor_symbol1)
                if not restrictor2_1:
                    continue

                restrictor_label1_1 = restrictor_data1["template"]
                restrictor_label2_1 = restrictor2_1["template"]
                kind_noun1_1 = restrictor_label1_1.replace("[1] is a ", "").strip()
                kind_noun2_1 = restrictor_label2_1.replace("[1] is a ", "").strip()
                art1_1 = "An" if kind_noun1_1[0].lower() in "aeiou" else "A"
                art2_1 = "An" if kind_noun2_1[0].lower() in "aeiou" else "A"

                for restrictor_symbol2, restrictor_data2 in kind_predicates1.items():
                    # get corresponding restrictor from lexicon2
                    restrictor2_2 = kind_predicates2.get(restrictor_symbol2)
                    if not restrictor2_2:
                        continue

                    restrictor_label1_2 = restrictor_data2["template"]
                    restrictor_label2_2 = restrictor2_2["template"]
                    kind_noun1_2 = restrictor_label1_2.replace("[1] is a ", "").strip()
                    kind_noun2_2 = restrictor_label2_2.replace("[1] is a ", "").strip()
                    art1_2 = "An" if kind_noun1_2[0].lower() in "aeiou" else "A"
                    art2_2 = "An" if kind_noun2_2[0].lower() in "aeiou" else "A"

                    soa1 = {
                        restrictor_symbol1: restrictor_label1_1,
                        restrictor_symbol2: restrictor_label1_2,
                        pred_symbol: pred_data["template"],
                    }
                    soa2 = {
                        restrictor_symbol1: restrictor_label2_1,
                        restrictor_symbol2: restrictor_label2_2,
                        pred_symbol: dyadic_predicates2[pred_symbol]["template"],
                    }

                    # all_all:    Every A R's every B     :: ∀x∀y((Ax ∧ By) → Rxy)
                    all_all1 = {
                        "sentence": f"Every {kind_noun1_1} {relation1} every {kind_noun1_2}.",
                        "type": "quantified",
                        "subtype": "all_all",
                        "soa": soa1,
                        "form": f"∀x∀y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) → {pred_symbol}xy)",
                        "language": self.language1,
                    }
                    self.add_entry(**all_all1)
                    sentence1_id = self.db.get_last_inserted_id()

                    all_all2 = {
                        "sentence": f"Every {kind_noun2_1} {relation2} every {kind_noun2_2}.",
                        "type": "quantified",
                        "subtype": "all_all",
                        "soa": soa2,
                        "form": f"∀x∀y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) → {pred_symbol}xy)",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**all_all2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # all_some:   Every A R's some B      :: ∀x(Ax → ∃y(By ∧ Rxy))
                    all_some1 = {
                        "sentence": f"Every {kind_noun1_1} {relation1} {art1_2.lower()} {kind_noun1_2}.",
                        "type": "quantified",
                        "subtype": "all_some",
                        "soa": soa1,
                        "form": f"∀x({restrictor_symbol1}x → ∃y({restrictor_symbol2}y ∧ {pred_symbol}xy))",
                        "language": self.language1,
                    }
                    self.add_entry(**all_some1)
                    sentence1_id = self.db.get_last_inserted_id()

                    all_some2 = {
                        "sentence": f"Every {kind_noun2_1} {relation2} {art2_2.lower()} {kind_noun2_2}.",
                        "type": "quantified",
                        "subtype": "all_some",
                        "soa": soa2,
                        "form": f"∀x({restrictor_symbol1}x → ∃y({restrictor_symbol2}y ∧ {pred_symbol}xy))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**all_some2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # some_all:   An A R's every B        :: ∃x(Ax ∧ ∀y(By → Rxy))
                    some_all1 = {
                        "sentence": f"{art1_1} {kind_noun1_1} {relation1} every {kind_noun1_2}.",
                        "type": "quantified",
                        "subtype": "some_all",
                        "soa": soa1,
                        "form": f"∃x({restrictor_symbol1}x ∧ ∀y({restrictor_symbol2}y → {pred_symbol}xy))",
                        "language": self.language1,
                    }
                    self.add_entry(**some_all1)
                    sentence1_id = self.db.get_last_inserted_id()

                    some_all2 = {
                        "sentence": f"{art2_1} {kind_noun2_1} {relation2} every {kind_noun2_2}.",
                        "type": "quantified",
                        "subtype": "some_all",
                        "soa": soa2,
                        "form": f"∃x({restrictor_symbol1}x ∧ ∀y({restrictor_symbol2}y → {pred_symbol}xy))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**some_all2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # some_some:  An A R's a B            :: ∃x∃y((Ax ∧ By) ∧ Rxy)
                    some_some1 = {
                        "sentence": f"{art1_1} {kind_noun1_1} {relation1} {art1_2.lower()} {kind_noun1_2}.",
                        "type": "quantified",
                        "subtype": "some_some",
                        "soa": soa1,
                        "form": f"∃x∃y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) ∧ {pred_symbol}xy)",
                        "language": self.language1,
                    }
                    self.add_entry(**some_some1)
                    sentence1_id = self.db.get_last_inserted_id()

                    some_some2 = {
                        "sentence": f"{art2_1} {kind_noun2_1} {relation2} {art2_2.lower()} {kind_noun2_2}.",
                        "type": "quantified",
                        "subtype": "some_some",
                        "soa": soa2,
                        "form": f"∃x∃y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) ∧ {pred_symbol}xy)",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**some_some2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # no_all:     No A R's every B        :: ¬∃x(Ax ∧ ∀y(By → Rxy))
                    no_all1 = {
                        "sentence": f"No {kind_noun1_1}s {relation1} every {kind_noun1_2}.",
                        "type": "quantified",
                        "subtype": "no_all",
                        "soa": soa1,
                        "form": f"¬∃x({restrictor_symbol1}x ∧ ∀y({restrictor_symbol2}y → {pred_symbol}xy))",
                        "language": self.language1,
                    }
                    self.add_entry(**no_all1)
                    sentence1_id = self.db.get_last_inserted_id()

                    no_all2 = {
                        "sentence": f"No {kind_noun2_1}s {relation2} every {kind_noun2_2}.",
                        "type": "quantified",
                        "subtype": "no_all",
                        "soa": soa2,
                        "form": f"¬∃x({restrictor_symbol1}x ∧ ∀y({restrictor_symbol2}y → {pred_symbol}xy))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**no_all2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # no_some:    No A R's a B            :: ¬∃x∃y((Ax ∧ By) ∧ Rxy)
                    no_some1 = {
                        "sentence": f"No {kind_noun1_1}s {relation1} {art1_2.lower()} {kind_noun1_2}.",
                        "type": "quantified",
                        "subtype": "no_some",
                        "soa": soa1,
                        "form": f"¬∃x∃y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) ∧ {pred_symbol}xy)",
                        "language": self.language1,
                    }
                    self.add_entry(**no_some1)
                    sentence1_id = self.db.get_last_inserted_id()

                    no_some2 = {
                        "sentence": f"No {kind_noun2_1}s {relation2} {art2_2.lower()} {kind_noun2_2}.",
                        "type": "quantified",
                        "subtype": "no_some",
                        "soa": soa2,
                        "form": f"¬∃x∃y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) ∧ {pred_symbol}xy)",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**no_some2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # rev_some_all: There is an A that every B Rs  :: ∃x(Ax∧∀y(By→Ryx))
                    rev_some_all1 = {
                        "sentence": f"There is {art1_1.lower()} {kind_noun1_1} that every {kind_noun1_2} {relation1}.",
                        "type": "quantified",
                        "subtype": "rev_some_all",
                        "soa": soa1,
                        "form": f"∃x({restrictor_symbol1}x∧∀y({restrictor_symbol2}y→{pred_symbol}yx))",
                        "language": self.language1,
                    }
                    self.add_entry(**rev_some_all1)
                    sentence1_id = self.db.get_last_inserted_id()

                    rev_some_all2 = {
                        "sentence": f"There is {art2_1.lower()} {kind_noun2_1} that every {kind_noun2_2} {relation2}.",
                        "type": "quantified",
                        "subtype": "rev_some_all",
                        "soa": soa2,
                        "form": f"∃x({restrictor_symbol1}x∧∀y({restrictor_symbol2}y→{pred_symbol}yx))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**rev_some_all2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # rev_no_all: There is not an A that every B Rs  :: ¬∃x(Ax∧∀y(By→Ryx))
                    rev_no_all1 = {
                        "sentence": f"There is not {art1_1.lower()} {kind_noun1_1} that every {kind_noun1_2} {relation1}.",
                        "type": "quantified",
                        "subtype": "rev_no_all",
                        "soa": soa1,
                        "form": f"¬∃x({restrictor_symbol1}x∧∀y({restrictor_symbol2}y→{pred_symbol}yx))",
                        "language": self.language1,
                    }
                    self.add_entry(**rev_no_all1)
                    sentence1_id = self.db.get_last_inserted_id()

                    rev_no_all2 = {
                        "sentence": f"There is not {art2_1.lower()} {kind_noun2_1} that every {kind_noun2_2} {relation2}.",
                        "type": "quantified",
                        "subtype": "rev_no_all",
                        "soa": soa2,
                        "form": f"¬∃x({restrictor_symbol1}x∧∀y({restrictor_symbol2}y→{pred_symbol}yx))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**rev_no_all2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                # some_self: An A Rs itself  :: ∃x(Ax ∧ Rxx)
                some_self1 = {
                    "sentence": f"{art1_1} {kind_noun1_1} {relation1} itself.",
                    "type": "quantified",
                    "subtype": "some_self",
                    "soa": {
                        restrictor_symbol1: restrictor_label1_1,
                        pred_symbol: pred_data["template"],
                    },
                    "form": f"∃x({restrictor_symbol1}x ∧ {pred_symbol}xx)",
                    "language": self.language1,
                }
                self.add_entry(**some_self1)
                sentence1_id = self.db.get_last_inserted_id()

                some_self2 = {
                    "sentence": f"{art2_1} {kind_noun2_1} {relation2} itself.",
                    "type": "quantified",
                    "subtype": "some_self",
                    "soa": {
                        restrictor_symbol1: restrictor_label2_1,
                        pred_symbol: dyadic_predicates2[pred_symbol]["template"],
                    },
                    "form": f"∃x({restrictor_symbol1}x ∧ {pred_symbol}xx)",
                    "counterpart_id": sentence1_id,
                    "language": self.language2,
                }
                self.add_entry(**some_self2)
                sentence2_id = self.db.get_last_inserted_id()
                self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

    def generate_reciprocal_quantified_sentences(self):
        """Generate reciprocal quantified sentences."""
        # generate sentences with reciprocal relationships
        kind_predicates1 = {
            k: v for k, v in self.lexicon1.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        kind_predicates2 = {
            k: v for k, v in self.lexicon2.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        dyadic_predicates1 = {}
        dyadic_predicates2 = {}
        for pred_symbol, pred_data in self.lexicon1.predicates.items():
            if pred_data["arity"] == 2:
                dyadic_predicates1[pred_symbol] = pred_data
                # get corresponding predicate from lexicon2
                pred2 = self.lexicon2.predicates.get(pred_symbol)
                if pred2:
                    dyadic_predicates2[pred_symbol] = pred2

        for pred_symbol, pred_data in dyadic_predicates1.items():
            if pred_symbol in kind_predicates1:
                continue
            relation1 = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
            relation2 = dyadic_predicates2[pred_symbol]["template"].replace("[1]", "").replace("[2]", "").strip()

            for restrictor_symbol1, restrictor_data1 in kind_predicates1.items():
                # get corresponding restrictor from lexicon2
                restrictor2_1 = kind_predicates2.get(restrictor_symbol1)
                if not restrictor2_1:
                    continue

                restrictor_label1_1 = restrictor_data1["template"]
                restrictor_label2_1 = restrictor2_1["template"]
                kind_noun1_1 = restrictor_label1_1.replace("[1] is a ", "").strip()
                kind_noun2_1 = restrictor_label2_1.replace("[1] is a ", "").strip()
                art1_1 = "An" if kind_noun1_1[0].lower() in "aeiou" else "A"
                art2_1 = "An" if kind_noun2_1[0].lower() in "aeiou" else "A"

                for restrictor_symbol2, restrictor_data2 in kind_predicates1.items():
                    # get corresponding restrictor from lexicon2
                    restrictor2_2 = kind_predicates2.get(restrictor_symbol2)
                    if not restrictor2_2:
                        continue

                    restrictor_label1_2 = restrictor_data2["template"]
                    restrictor_label2_2 = restrictor2_2["template"]
                    kind_noun1_2 = restrictor_label1_2.replace("[1] is a ", "").strip()
                    kind_noun2_2 = restrictor_label2_2.replace("[1] is a ", "").strip()
                    art1_2 = "An" if kind_noun1_2[0].lower() in "aeiou" else "A"
                    art2_2 = "An" if kind_noun2_2[0].lower() in "aeiou" else "A"

                    soa1 = {
                        restrictor_symbol1: restrictor_label1_1,
                        restrictor_symbol2: restrictor_label1_2,
                        pred_symbol: pred_data["template"],
                    }
                    soa2 = {
                        restrictor_symbol1: restrictor_label2_1,
                        restrictor_symbol2: restrictor_label2_2,
                        pred_symbol: dyadic_predicates2[pred_symbol]["template"],
                    }

                    # all_all_back: Every A Rs every B that Rs it :: ∀x(Ax→∀y((By∧Ryx)→Rxy))
                    all_all_back1 = {
                        "sentence": f"Every {kind_noun1_1} {relation1} every {kind_noun1_2} that {relation1} it.",
                        "type": "quantified",
                        "subtype": "all_all_back",
                        "soa": soa1,
                        "form": f"∀x({restrictor_symbol1}x→∀y(({restrictor_symbol2}y∧{pred_symbol}yx)→{pred_symbol}xy))",
                        "language": self.language1,
                    }
                    self.add_entry(**all_all_back1)
                    sentence1_id = self.db.get_last_inserted_id()

                    all_all_back2 = {
                        "sentence": f"Every {kind_noun2_1} {relation2} every {kind_noun2_2} that {relation2} it.",
                        "type": "quantified",
                        "subtype": "all_all_back",
                        "soa": soa2,
                        "form": f"∀x({restrictor_symbol1}x→∀y(({restrictor_symbol2}y∧{pred_symbol}yx)→{pred_symbol}xy))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**all_all_back2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # some_all_back: Some A Rs every B that Rs it :: ∃x(Ax∧∀y((By∧Ryx)→Rxy))
                    some_all_back1 = {
                        "sentence": f"{art1_1} {kind_noun1_1} {relation1} every {kind_noun1_2} that {relation1} it.",
                        "type": "quantified",
                        "subtype": "some_all_back",
                        "soa": soa1,
                        "form": f"∃x({restrictor_symbol1}x∧∀y(({restrictor_symbol2}y∧{pred_symbol}yx)→{pred_symbol}xy))",
                        "language": self.language1,
                    }
                    self.add_entry(**some_all_back1)
                    sentence1_id = self.db.get_last_inserted_id()

                    some_all_back2 = {
                        "sentence": f"{art2_1} {kind_noun2_1} {relation2} every {kind_noun2_2} that {relation2} it.",
                        "type": "quantified",
                        "subtype": "some_all_back",
                        "soa": soa2,
                        "form": f"∃x({restrictor_symbol1}x∧∀y(({restrictor_symbol2}y∧{pred_symbol}yx)→{pred_symbol}xy))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**some_all_back2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # some_some_back: Some A Rs some B that Rs it :: ∃x∃y((Ax∧By)∧(Rxy∧Ryx))
                    some_some_back1 = {
                        "sentence": f"{art1_1} {kind_noun1_1} {relation1} {art1_2.lower()} {kind_noun1_2} that {relation1} it.",
                        "type": "quantified",
                        "subtype": "some_some_back",
                        "soa": soa1,
                        "form": f"∃x∃y(({restrictor_symbol1}x∧{restrictor_symbol2}y)∧({pred_symbol}xy∧{pred_symbol}yx))",
                        "language": self.language1,
                    }
                    self.add_entry(**some_some_back1)
                    sentence1_id = self.db.get_last_inserted_id()

                    some_some_back2 = {
                        "sentence": f"{art2_1} {kind_noun2_1} {relation2} {art2_2.lower()} {kind_noun2_2} that {relation2} it.",
                        "type": "quantified",
                        "subtype": "some_some_back",
                        "soa": soa2,
                        "form": f"∃x∃y(({restrictor_symbol1}x∧{restrictor_symbol2}y)∧({pred_symbol}xy∧{pred_symbol}yx))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**some_some_back2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # all_some_back: Every A Rs some B that Rs it :: ∀x(Ax→∃y(By∧(Rxy∧Ryx)))
                    all_some_back1 = {
                        "sentence": f"Every {kind_noun1_1} {relation1} {art1_2.lower()} {kind_noun1_2} that {relation1} it.",
                        "type": "quantified",
                        "subtype": "all_some_back",
                        "soa": soa1,
                        "form": f"∀x({restrictor_symbol1}x→∃y({restrictor_symbol2}y∧({pred_symbol}xy∧{pred_symbol}yx)))",
                        "language": self.language1,
                    }
                    self.add_entry(**all_some_back1)
                    sentence1_id = self.db.get_last_inserted_id()

                    all_some_back2 = {
                        "sentence": f"Every {kind_noun2_1} {relation2} {art2_2.lower()} {kind_noun2_2} that {relation2} it.",
                        "type": "quantified",
                        "subtype": "all_some_back",
                        "soa": soa2,
                        "form": f"∀x({restrictor_symbol1}x→∃y({restrictor_symbol2}y∧({pred_symbol}xy∧{pred_symbol}yx)))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**all_some_back2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # no_some_back: No A Rs some B that Rs it :: ¬∃x∃y((Ax∧By)∧(Rxy∧Ryx))
                    no_some_back1 = {
                        "sentence": f"No {kind_noun1_1}s {relation1} {art1_2.lower()} {kind_noun1_2} that {relation1} it.",
                        "type": "quantified",
                        "subtype": "no_some_back",
                        "soa": soa1,
                        "form": f"¬∃x∃y(({restrictor_symbol1}x∧{restrictor_symbol2}y)∧({pred_symbol}xy∧{pred_symbol}yx))",
                        "language": self.language1,
                    }
                    self.add_entry(**no_some_back1)
                    sentence1_id = self.db.get_last_inserted_id()

                    no_some_back2 = {
                        "sentence": f"No {kind_noun2_1}s {relation2} {art2_2.lower()} {kind_noun2_2} that {relation2} it.",
                        "type": "quantified",
                        "subtype": "no_some_back",
                        "soa": soa2,
                        "form": f"¬∃x∃y(({restrictor_symbol1}x∧{restrictor_symbol2}y)∧({pred_symbol}xy∧{pred_symbol}yx))",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**no_some_back2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

    def generate_complex_dyadic_sentences(self):
        """Generate sentences with three quantifiers and dyadic predicates."""
        kind_predicates1 = {
            k: v for k, v in self.lexicon1.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        kind_predicates2 = {
            k: v for k, v in self.lexicon2.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        dyadic_predicates1 = {}
        dyadic_predicates2 = {}
        for pred_symbol, pred_data in self.lexicon1.predicates.items():
            if pred_data["arity"] == 2:
                dyadic_predicates1[pred_symbol] = pred_data
                # get corresponding predicate from lexicon2
                pred2 = self.lexicon2.predicates.get(pred_symbol)
                if pred2:
                    dyadic_predicates2[pred_symbol] = pred2

        for pred_symbol, pred_data in dyadic_predicates1.items():
            if pred_symbol in kind_predicates1:
                continue
            relation1 = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
            relation2 = dyadic_predicates2[pred_symbol]["template"].replace("[1]", "").replace("[2]", "").strip()

            for restrictor_symbol1, restrictor_data1 in kind_predicates1.items():
                # get corresponding restrictor from lexicon2
                restrictor2_1 = kind_predicates2.get(restrictor_symbol1)
                if not restrictor2_1:
                    continue

                restrictor_label1_1 = restrictor_data1["template"]
                restrictor_label2_1 = restrictor2_1["template"]
                kind_noun1_1 = restrictor_label1_1.replace("[1] is a ", "").strip()
                kind_noun2_1 = restrictor_label2_1.replace("[1] is a ", "").strip()
                art1_1 = "An" if kind_noun1_1[0].lower() in "aeiou" else "A"
                art2_1 = "An" if kind_noun2_1[0].lower() in "aeiou" else "A"

                for restrictor_symbol2, restrictor_data2 in kind_predicates1.items():
                    # get corresponding restrictor from lexicon2
                    restrictor2_2 = kind_predicates2.get(restrictor_symbol2)
                    if not restrictor2_2:
                        continue

                    restrictor_label1_2 = restrictor_data2["template"]
                    restrictor_label2_2 = restrictor2_2["template"]
                    kind_noun1_2 = restrictor_label1_2.replace("[1] is a ", "").strip()
                    kind_noun2_2 = restrictor_label2_2.replace("[1] is a ", "").strip()
                    art1_2 = "An" if kind_noun1_2[0].lower() in "aeiou" else "A"
                    art2_2 = "An" if kind_noun2_2[0].lower() in "aeiou" else "A"

                    for restrictor_symbol3, restrictor_data3 in kind_predicates1.items():
                        # get corresponding restrictor from lexicon2
                        restrictor2_3 = kind_predicates2.get(restrictor_symbol3)
                        if not restrictor2_3:
                            continue

                        restrictor_label1_3 = restrictor_data3["template"]
                        restrictor_label2_3 = restrictor2_3["template"]
                        kind_noun1_3 = restrictor_label1_3.replace("[1] is a ", "").strip()
                        kind_noun2_3 = restrictor_label2_3.replace("[1] is a ", "").strip()
                        art1_3 = "An" if kind_noun1_3[0].lower() in "aeiou" else "A"
                        art2_3 = "An" if kind_noun2_3[0].lower() in "aeiou" else "A"

                        soa1 = {
                            restrictor_symbol1: restrictor_label1_1,
                            restrictor_symbol2: restrictor_label1_2,
                            restrictor_symbol3: restrictor_label1_3,
                            pred_symbol: pred_data["template"],
                        }
                        soa2 = {
                            restrictor_symbol1: restrictor_label2_1,
                            restrictor_symbol2: restrictor_label2_2,
                            restrictor_symbol3: restrictor_label2_3,
                            pred_symbol: dyadic_predicates2[pred_symbol]["template"],
                        }

                        # all_all_all: Every A Rs every B that Rs every C :: ∀x(Ax→∀y((By∧∀z(Cz→Ryz))→Rxy))
                        all_all_all1 = {
                            "sentence": (
                                f"Every {kind_noun1_1} {relation1} every {kind_noun1_2} that {relation1} every {kind_noun1_3}."
                            ),
                            "type": "quantified",
                            "subtype": "all_all_all",
                            "soa": soa1,
                            "form": f"∀x({restrictor_symbol1}x→∀y(({restrictor_symbol2}y∧∀z({restrictor_symbol3}z→{pred_symbol}yz))→{pred_symbol}xy))",
                            "language": self.language1,
                        }
                        self.add_entry(**all_all_all1)
                        sentence1_id = self.db.get_last_inserted_id()

                        all_all_all2 = {
                            "sentence": (
                                f"Every {kind_noun2_1} {relation2} every {kind_noun2_2} that {relation2} every {kind_noun2_3}."
                            ),
                            "type": "quantified",
                            "subtype": "all_all_all",
                            "soa": soa2,
                            "form": f"∀x({restrictor_symbol1}x→∀y(({restrictor_symbol2}y∧∀z({restrictor_symbol3}z→{pred_symbol}yz))→{pred_symbol}xy))",
                            "counterpart_id": sentence1_id,
                            "language": self.language2,
                        }
                        self.add_entry(**all_all_all2)
                        sentence2_id = self.db.get_last_inserted_id()
                        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                        # some_some_some: Some A Rs some B that Rs some C :: ∃x∃y∃z(((Ax∧By)∧Cz)∧(Rxy∧Ryz))
                        some_some_some1 = {
                            "sentence": (
                                f"{art1_1} {kind_noun1_1} {relation1} {art1_2.lower()} {kind_noun1_2} that {relation1} {art1_3.lower()} {kind_noun1_3}."
                            ),
                            "type": "quantified",
                            "subtype": "some_some_some",
                            "soa": soa1,
                            "form": f"∃x∃y∃z((({restrictor_symbol1}x∧{restrictor_symbol2}y)∧{restrictor_symbol3}z)∧({pred_symbol}xy∧{pred_symbol}yz))",
                            "language": self.language1,
                        }
                        self.add_entry(**some_some_some1)
                        sentence1_id = self.db.get_last_inserted_id()

                        some_some_some2 = {
                            "sentence": (
                                f"{art2_1} {kind_noun2_1} {relation2} {art2_2.lower()} {kind_noun2_2} that {relation2} {art2_3.lower()} {kind_noun2_3}."
                            ),
                            "type": "quantified",
                            "subtype": "some_some_some",
                            "soa": soa2,
                            "form": f"∃x∃y∃z((({restrictor_symbol1}x∧{restrictor_symbol2}y)∧{restrictor_symbol3}z)∧({pred_symbol}xy∧{pred_symbol}yz))",
                            "counterpart_id": sentence1_id,
                            "language": self.language2,
                        }
                        self.add_entry(**some_some_some2)
                        sentence2_id = self.db.get_last_inserted_id()
                        self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

    def generate_name_quantified_sentences(self):
        """Generate name quantified sentences."""
        kind_predicates1 = {
            k: v for k, v in self.lexicon1.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        kind_predicates2 = {
            k: v for k, v in self.lexicon2.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
        }
        dyadic_predicates1 = {}
        dyadic_predicates2 = {}
        for pred_symbol, pred_data in self.lexicon1.predicates.items():
            if pred_data["arity"] == 2:
                dyadic_predicates1[pred_symbol] = pred_data
                # get corresponding predicate from lexicon2
                pred2 = self.lexicon2.predicates.get(pred_symbol)
                if pred2:
                    dyadic_predicates2[pred_symbol] = pred2

        for pred_symbol, pred_data in dyadic_predicates1.items():
            if pred_symbol in kind_predicates1:
                continue
            relation1 = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
            relation2 = dyadic_predicates2[pred_symbol]["template"].replace("[1]", "").replace("[2]", "").strip()

            for restrictor_symbol, restrictor_data in kind_predicates1.items():
                # get corresponding restrictor from lexicon2
                restrictor2 = kind_predicates2.get(restrictor_symbol)
                if not restrictor2:
                    continue

                restrictor_label1 = restrictor_data["template"]
                restrictor_label2 = restrictor2["template"]
                kind_noun1 = restrictor_label1.replace("[1] is a ", "").strip()
                kind_noun2 = restrictor_label2.replace("[1] is a ", "").strip()
                art1 = "An" if kind_noun1[0].lower() in "aeiou" else "A"
                art2 = "An" if kind_noun2[0].lower() in "aeiou" else "A"

                for name_sym, name_data in self.lexicon1.names.items():
                    name1 = name_data["name"]
                    name2 = self.lexicon2.get_name(name_sym)

                    soa1 = {
                        restrictor_symbol: restrictor_label1,
                        pred_symbol: pred_data["template"],
                        name_sym: name1,
                    }
                    soa2 = {
                        restrictor_symbol: restrictor_label2,
                        pred_symbol: dyadic_predicates2[pred_symbol]["template"],
                        name_sym: name2,
                    }

                    # name_all: John R's every A  :: ∀x(Ax→Rjx)
                    name_all1 = {
                        "sentence": f"{name1} {relation1} every {kind_noun1}.",
                        "type": "quantified",
                        "subtype": "name_all",
                        "soa": soa1,
                        "form": f"∀x({restrictor_symbol}x→{pred_symbol}{name_sym}x)",
                        "language": self.language1,
                    }
                    self.add_entry(**name_all1)
                    sentence1_id = self.db.get_last_inserted_id()

                    name_all2 = {
                        "sentence": f"{name2} {relation2} every {kind_noun2}.",
                        "type": "quantified",
                        "subtype": "name_all",
                        "soa": soa2,
                        "form": f"∀x({restrictor_symbol}x→{pred_symbol}{name_sym}x)",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**name_all2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

                    # name_some: John R's a A  :: ∃x(Ax∧Rjx)
                    name_some1 = {
                        "sentence": f"{name1} {relation1} {art1.lower()} {kind_noun1}.",
                        "type": "quantified",
                        "subtype": "name_some",
                        "soa": soa1,
                        "form": f"∃x({restrictor_symbol}x∧{pred_symbol}{name_sym}x)",
                        "language": self.language1,
                    }
                    self.add_entry(**name_some1)
                    sentence1_id = self.db.get_last_inserted_id()

                    name_some2 = {
                        "sentence": f"{name2} {relation2} {art2.lower()} {kind_noun2}.",
                        "type": "quantified",
                        "subtype": "name_some",
                        "soa": soa2,
                        "form": f"∃x({restrictor_symbol}x∧{pred_symbol}{name_sym}x)",
                        "counterpart_id": sentence1_id,
                        "language": self.language2,
                    }
                    self.add_entry(**name_some2)
                    sentence2_id = self.db.get_last_inserted_id()
                    self.db.update_sentence_counterpart(sentence1_id, sentence2_id)

    # ===== Compound Sentence Generation Methods =====

    def generate_conjunctions(self):
        """Generate conjunction sentences."""
        sampled_entries = self._sample_base_entries()

        for e1, e2 in itertools.combinations(sampled_entries, 2):
            if e1["counterpart_id"] is None or e2["counterpart_id"] is None:
                continue

            # prioritize combinations that mix quantifier types
            if (
                e1["type"] == "quantified"
                and e2["type"] == "quantified"
                and e1["form"].startswith("∀") != e2["form"].startswith("∀")
            ):
                self._add_and(e1, e2)
                continue

            # include combinations with binary predicates
            if any("∃" in e["form"] and "∧" in e["form"] for e in [e1, e2]):
                self._add_and(e1, e2)
                continue

            # include some non-quantified combinations
            if random.random() < 0.5:  # 50% chance to include non-quantified combinations
                self._add_and(e1, e2)

    def generate_disjunctions(self):
        """Generate disjunction sentences."""
        sampled_entries = self._sample_base_entries()

        for e1, e2 in itertools.combinations(sampled_entries, 2):
            if e1["counterpart_id"] is None or e2["counterpart_id"] is None:
                continue

            # prioritize combinations that create interesting logical structures
            if e1["type"] == "quantified" and e2["type"] == "quantified" and "→" in e1["form"] and "→" in e2["form"]:
                self._add_or(e1, e2)
                continue

            # include combinations that mix universal and existential
            if (
                e1["type"] == "quantified"
                and e2["type"] == "quantified"
                and e1["form"].startswith("∀") != e2["form"].startswith("∀")
            ):
                self._add_or(e1, e2)
                continue

            # include some non-quantified combinations
            if random.random() < 0.5:  # 50% chance to include non-quantified combinations
                self._add_or(e1, e2)

    def generate_conditionals(self):
        """Generate conditional sentences."""
        sampled_entries = self._sample_base_entries()

        for e1, e2 in itertools.combinations(sampled_entries, 2):
            if e1["counterpart_id"] is None or e2["counterpart_id"] is None:
                continue

            # don't use only_restrictor quantified sentences
            if e1["subtype"] == "only_restrictor" or e2["subtype"] == "only_restrictor":
                continue

            # prioritize combinations that create nested conditionals
            if e1["type"] == "quantified" and "→" in e1["form"] and e2["type"] == "quantified" and "→" in e2["form"]:
                self._add_conditional(e1, e2)
                continue

            # include combinations that mix universal and existential
            if (
                e1["type"] == "quantified"
                and e2["type"] == "quantified"
                and e1["form"].startswith("∀") != e2["form"].startswith("∀")
            ):
                self._add_conditional(e1, e2)
                continue

            # include some non-quantified combinations
            if random.random() < 0.4:  # 40% chance to include non-quantified combinations
                self._add_conditional(e1, e2)

    def generate_biconditionals(self):
        """Generate biconditional sentences."""
        sampled_entries = self._sample_base_entries()

        for e1, e2 in itertools.combinations(sampled_entries, 2):
            if e1["counterpart_id"] is None or e2["counterpart_id"] is None:
                continue

            # skip only_restrictor quantified sentences
            if e1["subtype"] == "only_restrictor" or e2["subtype"] == "only_restrictor":
                continue

            # prioritize combinations that create interesting equivalences
            if (
                e1["type"] == "quantified"
                and e2["type"] == "quantified"
                and e1["form"].startswith("∀") != e2["form"].startswith("∀")
            ):
                self._add_iff(e1, e2)
                continue

            # include combinations with binary predicates
            if any("∃" in e["form"] and "∧" in e["form"] for e in [e1, e2]):
                self._add_iff(e1, e2)
                continue

            # include some non-quantified combinations
            if random.random() < 0.3:  # 30% chance to include non-quantified combinations
                self._add_iff(e1, e2)

    def generate_nested_conditionals(self):
        """Generate nested conditional sentences."""
        sampled_entries = self._sample_base_entries()

        for e1, e2, e3 in itertools.combinations(sampled_entries, 3):
            if e1["counterpart_id"] is None or e2["counterpart_id"] is None or e3["counterpart_id"] is None:
                continue

            # skip if all entries are of the same type
            if e1["type"] == e2["type"] == e3["type"]:
                continue
            if (
                e1["subtype"] == "only_restrictor"
                or e2["subtype"] == "only_restrictor"
                or e3["subtype"] == "only_restrictor"
            ):
                continue
            # only create if at least two entries are quantified
            if sum(e["type"] == "quantified" for e in [e1, e2, e3]) >= 2:
                self._add_if_then_only_if(e1, e2, e3)
            elif random.random() < 0.05:  # 5% chance to include non-quantified combinations
                self._add_if_then_only_if(e1, e2, e3)

    # ===== Compound Sentence Helper Methods =====

    def _add_and(self, e1, e2):
        """Add conjunction sentences."""
        # get counterparts of e1 and e2
        counterpart_id1 = e1["counterpart_id"]
        counterpart_id2 = e2["counterpart_id"]
        if counterpart_id1 is None or counterpart_id2 is None:
            logger.warning(f"No counterpart found for {e1['sentence']} or {e2['sentence']}")
            return
        e1_counterpart = self.db.get_sentence_where(id=counterpart_id1)[0]
        e2_counterpart = self.db.get_sentence_where(id=counterpart_id2)[0]

        # create English conjunction
        s1 = self._capitalize(self._lowercase_except_names(e1["sentence"].rstrip("."), self.lexicon1))
        s2 = self._lowercase_except_names(e2["sentence"].rstrip("."), self.lexicon1)
        soa1 = {**e1["soa"], **e2["soa"]}

        # add English conjunction
        self.add_entry(
            sentence=f"{s1} and {s2}.",
            type="conjunction",
            subtype="simple",
            soa=soa1,
            form=f"({e1['form']}∧{e2['form']})",
            language=self.language1,
        )
        conj1_id = self.db.get_last_inserted_id()

        # create Carroll conjunction
        s1_c = self._capitalize(self._lowercase_except_names(e1_counterpart["sentence"].rstrip("."), self.lexicon2))
        s2_c = self._lowercase_except_names(e2_counterpart["sentence"].rstrip("."), self.lexicon2)
        soa2 = {**e1_counterpart["soa"], **e2_counterpart["soa"]}

        # add Carroll conjunction with English as counterpart
        self.add_entry(
            sentence=f"{s1_c} and {s2_c}.",
            type="conjunction",
            subtype="simple",
            soa=soa2,
            form=f"({e1['form']}∧{e2['form']})",  # Use same logical form
            counterpart_id=conj1_id,
            language=self.language2,
        )
        conj2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(conj1_id, conj2_id)

        # add contrastive conjunctions if applicable
        if e2["type"] != "quantified":
            # English contrastive
            self.add_entry(
                sentence=f"{s1} but it is not the case that {s2}.",
                type="conjunction",
                subtype="contrastive",
                soa=soa1,
                form=f"({e1['form']}∧¬{e2['form']})",
                language=self.language1,
            )
            contrast1_id = self.db.get_last_inserted_id()

            # Carroll contrastive
            self.add_entry(
                sentence=f"{s1_c} but it is not the case that {s2_c}.",
                type="conjunction",
                subtype="contrastive",
                soa=soa2,
                form=f"({e1['form']}∧¬{e2['form']})",  # Use same logical form
                counterpart_id=contrast1_id,
                language=self.language2,
            )
            contrast2_id = self.db.get_last_inserted_id()

            # update counterpart relationship
            self.db.update_sentence_counterpart(contrast1_id, contrast2_id)

    def _add_or(self, e1, e2):
        """Add disjunction sentences."""
        # get counterparts of e1 and e2
        counterpart_id1 = e1["counterpart_id"]
        counterpart_id2 = e2["counterpart_id"]
        if counterpart_id1 is None or counterpart_id2 is None:
            logger.warning(f"No counterpart found for {e1['sentence']} or {e2['sentence']}")
            return
        e1_counterpart = self.db.get_sentence_where(id=counterpart_id1)[0]
        e2_counterpart = self.db.get_sentence_where(id=counterpart_id2)[0]

        # create English disjunction
        s1 = self._capitalize(self._lowercase_except_names(e1["sentence"].rstrip("."), self.lexicon1))
        s2 = self._lowercase_except_names(e2["sentence"].rstrip("."), self.lexicon1)
        soa1 = {**e1["soa"], **e2["soa"]}

        # add English simple disjunction
        self.add_entry(
            sentence=f"{s1} or {s2}.",
            type="disjunction",
            subtype="simple",
            soa=soa1,
            form=f"({e1['form']}∨{e2['form']})",
            language=self.language1,
        )
        disj1_id = self.db.get_last_inserted_id()

        # create Carroll disjunction
        s1_c = self._capitalize(self._lowercase_except_names(e1_counterpart["sentence"].rstrip("."), self.lexicon2))
        s2_c = self._lowercase_except_names(e2_counterpart["sentence"].rstrip("."), self.lexicon2)
        soa2 = {**e1_counterpart["soa"], **e2_counterpart["soa"]}

        # add Carroll simple disjunction
        self.add_entry(
            sentence=f"{s1_c} or {s2_c}.",
            type="disjunction",
            subtype="simple",
            soa=soa2,
            form=f"({e1['form']}∨{e2['form']})",
            counterpart_id=disj1_id,
            language=self.language2,
        )
        disj2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(disj1_id, disj2_id)

        # add English "unless" disjunction
        self.add_entry(
            sentence=f"{s1} unless {s2}.",
            type="disjunction",
            subtype="unless",
            soa=soa1,
            form=f"(¬{e2['form']}→{e1['form']})",
            language=self.language1,
        )
        unless1_id = self.db.get_last_inserted_id()

        # add Carroll "unless" disjunction
        self.add_entry(
            sentence=f"{s1_c} unless {s2_c}.",
            type="disjunction",
            subtype="unless",
            soa=soa2,
            form=f"(¬{e2['form']}→{e1['form']})",  # Use same logical form
            counterpart_id=unless1_id,
            language=self.language2,
        )
        unless2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(unless1_id, unless2_id)

        # add negated disjunct if applicable
        if e2["type"] != "quantified":
            # English negated disjunct
            self.add_entry(
                sentence=f"{s1} or it is not the case that {s2}.",
                type="disjunction",
                subtype="negated_disjunct",
                soa=soa1,
                form=f"({e1['form']}∨¬{e2['form']})",
                language=self.language1,
            )
            neg1_id = self.db.get_last_inserted_id()

            # Carroll negated disjunct
            self.add_entry(
                sentence=f"{s1_c} or it is not the case that {s2_c}.",
                type="disjunction",
                subtype="negated_disjunct",
                soa=soa2,
                form=f"({e1['form']}∨¬{e2['form']})",  # Use same logical form
                counterpart_id=neg1_id,
                language=self.language2,
            )
            neg2_id = self.db.get_last_inserted_id()

            # update counterpart relationship
            self.db.update_sentence_counterpart(neg1_id, neg2_id)

    def _add_conditional(self, e1, e2):
        """Add conditional sentences."""
        # get counterparts of e1 and e2
        counterpart_id1 = e1["counterpart_id"]
        counterpart_id2 = e2["counterpart_id"]
        if counterpart_id1 is None or counterpart_id2 is None:
            logger.warning(f"No counterpart found for {e1['sentence']} or {e2['sentence']}")
            return

        # get counterparts using get_sentence_where
        e1_counterpart = self.db.get_sentence_where(id=counterpart_id1)[0]
        e2_counterpart = self.db.get_sentence_where(id=counterpart_id2)[0]

        # create English conditional
        s1 = self._lowercase_except_names(e1["sentence"].rstrip("."), self.lexicon1)
        s2 = self._lowercase_except_names(e2["sentence"].rstrip("."), self.lexicon1)
        soa1 = {**e1["soa"], **e2["soa"]}

        # add English if-then
        self.add_entry(
            sentence=f"If {s1}, then {s2}.",
            type="conditional",
            subtype="if_then",
            soa=soa1,
            form=f"({e1['form']}→{e2['form']})",
            language=self.language1,
        )
        if_then1_id = self.db.get_last_inserted_id()

        # create Carroll conditional
        s1_c = self._lowercase_except_names(e1_counterpart["sentence"].rstrip("."), self.lexicon2)
        s2_c = self._lowercase_except_names(e2_counterpart["sentence"].rstrip("."), self.lexicon2)
        soa2 = {**e1_counterpart["soa"], **e2_counterpart["soa"]}

        # add Carroll if-then
        self.add_entry(
            sentence=f"If {s1_c}, then {s2_c}.",
            type="conditional",
            subtype="if_then",
            soa=soa2,
            form=f"({e1['form']}→{e2['form']})",
            counterpart_id=if_then1_id,
            language=self.language2,
        )
        if_then2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(if_then1_id, if_then2_id)

        # add "only if" conditionals
        s1 = self._capitalize(s1)  # Capitalize for "only if" form
        s1_c = self._capitalize(s1_c)

        # English only-if
        self.add_entry(
            sentence=f"{s1} only if {s2}.",
            type="conditional",
            subtype="only_if",
            soa=soa1,
            form=f"({e1['form']}→{e2['form']})",
            language=self.language1,
        )
        only_if1_id = self.db.get_last_inserted_id()

        # Carroll only-if
        self.add_entry(
            sentence=f"{s1_c} only if {s2_c}.",
            type="conditional",
            subtype="only_if",
            soa=soa2,
            form=f"({e1['form']}→{e2['form']})",
            counterpart_id=only_if1_id,
            language=self.language2,
        )
        only_if2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(only_if1_id, only_if2_id)

    def _add_iff(self, e1, e2):
        """Add biconditional sentences."""
        # get counterparts of e1 and e2
        counterpart_id1 = e1["counterpart_id"]
        counterpart_id2 = e2["counterpart_id"]
        if counterpart_id1 is None or counterpart_id2 is None:
            logger.warning(f"No counterpart found for {e1['sentence']} or {e2['sentence']}")
            return

        # get counterparts using get_sentence_where
        e1_counterpart = self.db.get_sentence_where(id=counterpart_id1)[0]
        e2_counterpart = self.db.get_sentence_where(id=counterpart_id2)[0]

        # create English biconditional
        s1 = self._capitalize(self._lowercase_except_names(e1["sentence"].rstrip("."), self.lexicon1))
        s2 = self._lowercase_except_names(e2["sentence"].rstrip("."), self.lexicon1)
        soa1 = {**e1["soa"], **e2["soa"]}

        # randomly use "if and only if" or "just in case"
        flip_coin = random.choice([True, False])
        iff = "if and only if" if flip_coin else "just in case"
        subtype = "if_and_only_if" if flip_coin else "just_in_case"

        # add English biconditional
        self.add_entry(
            sentence=f"{s1} {iff} {s2}.",
            type="biconditional",
            subtype=subtype,
            soa=soa1,
            form=f"({e1['form']}↔{e2['form']})",
            language=self.language1,
        )
        bicond1_id = self.db.get_last_inserted_id()

        # create Carroll biconditional
        s1_c = self._capitalize(self._lowercase_except_names(e1_counterpart["sentence"].rstrip("."), self.lexicon2))
        s2_c = self._lowercase_except_names(e2_counterpart["sentence"].rstrip("."), self.lexicon2)
        soa2 = {**e1_counterpart["soa"], **e2_counterpart["soa"]}

        # add Carroll biconditional
        self.add_entry(
            sentence=f"{s1_c} {iff} {s2_c}.",
            type="biconditional",
            subtype=subtype,
            soa=soa2,
            form=f"({e1['form']}↔{e2['form']})",  # Use same logical form
            counterpart_id=bicond1_id,
            language=self.language2,
        )
        bicond2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(bicond1_id, bicond2_id)

    def _add_if_then_only_if(self, e1, e2, e3):
        """Add nested conditional sentences."""
        # get counterparts of e1, e2, and e3
        counterpart_id1 = e1["counterpart_id"]
        counterpart_id2 = e2["counterpart_id"]
        counterpart_id3 = e3["counterpart_id"]
        if counterpart_id1 is None or counterpart_id2 is None or counterpart_id3 is None:
            logger.warning("No counterpart found for one of the sentences")
            return
        e1_counterpart = self.db.get_sentence_where(id=counterpart_id1)[0]
        e2_counterpart = self.db.get_sentence_where(id=counterpart_id2)[0]
        e3_counterpart = self.db.get_sentence_where(id=counterpart_id3)[0]

        # create English nested conditional
        s1 = self._lowercase_except_names(e1["sentence"].rstrip("."), self.lexicon1)
        s2 = self._lowercase_except_names(e2["sentence"].rstrip("."), self.lexicon1)
        s3 = self._lowercase_except_names(e3["sentence"].rstrip("."), self.lexicon1)
        soa1 = {**e1["soa"], **e2["soa"], **e3["soa"]}

        # add English nested conditional
        self.add_entry(
            sentence=f"If {s1}, then {s2} only if {s3}.",
            type="conditional",
            subtype="nested",
            soa=soa1,
            form=f"({e1['form']}→({e2['form']}→{e3['form']}))",
            language=self.language1,
        )
        nested1_id = self.db.get_last_inserted_id()

        # create Carroll nested conditional
        s1_c = self._lowercase_except_names(e1_counterpart["sentence"].rstrip("."), self.lexicon2)
        s2_c = self._lowercase_except_names(e2_counterpart["sentence"].rstrip("."), self.lexicon2)
        s3_c = self._lowercase_except_names(e3_counterpart["sentence"].rstrip("."), self.lexicon2)
        soa2 = {**e1_counterpart["soa"], **e2_counterpart["soa"], **e3_counterpart["soa"]}

        # add Carroll nested conditional
        self.add_entry(
            sentence=f"If {s1_c}, then {s2_c} only if {s3_c}.",
            type="conditional",
            subtype="nested",
            soa=soa2,
            form=f"({e1['form']}→({e2['form']}→{e3['form']}))",  # Use same logical form
            counterpart_id=nested1_id,
            language=self.language2,
        )
        nested2_id = self.db.get_last_inserted_id()

        # update counterpart relationship
        self.db.update_sentence_counterpart(nested1_id, nested2_id)

    def _sample_base_entries(self, ratio=0.1):
        """Sample a random subset (~10%) of base entries."""
        entries = self.get_base_entries(language=self.language1)
        sample_size = int(len(entries) * ratio)
        sampled_entries = random.sample(entries, sample_size)
        return sampled_entries

    def _sample_base_entries_by_type(self, ratio=0.1):
        """Sample a random subset (~10%) of base entries by type."""
        entries = self.get_base_entries(language=self.language1)
        sample_size = int(len(entries) * ratio)
        sampled_entries = random.sample(entries, sample_size)


if __name__ == "__main__":
    from Syntax.carroll_lexicon import CarrollLexicon
    from Syntax.english_lexicon import EnglishLexicon

    english_lexicon = EnglishLexicon()
    carroll_lexicon = CarrollLexicon()
    generator = SentenceGenerator(english_lexicon, carroll_lexicon)

    def generate_all_sentences():
        """Generate all sentence types."""
        # print("Generating domain constraint...")
        # generator.generate_domain_constraint()

        # print("Generating atomic sentences...")
        # generator.generate_atomic_sentences()

        # print("Generating simple quantified sentences...")
        # generator.generate_simple_quantified_sentences()

        # print("Generating multiply quantified sentences...")
        # generator.generate_multiply_quantified_sentences()

        # print("Generating reciprocal quantified sentences...")
        # generator.generate_reciprocal_quantified_sentences()

        # print("Generating complex dyadic sentences...")
        # generator.generate_complex_dyadic_sentences()

        # print("Generating name quantified sentences...")
        # generator.generate_name_quantified_sentences()

        print("Generating compound sentences...")
        generator.generate_conjunctions()
        generator.generate_disjunctions()
        generator.generate_conditionals()
        generator.generate_biconditionals()
        # generator.generate_nested_conditionals()

    def create_samples(n=2):
        """Create a markdown file with samples of each sentence type."""
        with open("sentence_samples.md", "w") as f:
            f.write("# LogicSkills Sentence Samples\n\n")
            f.write(
                "This file contains samples of all sentence types generated by the LogicSkills sentence generator.\n\n"
            )

            def write_section(sentence_type, subtype=None):
                kwargs = {"type": sentence_type}
                if subtype:
                    kwargs["subtype"] = subtype

                # Get English sentences
                kwargs["language"] = generator.language1
                english_sentences = generator.db.get_sentence_where(**kwargs)

                # Filter to only include sentences that have counterparts
                english_sentences = [s for s in english_sentences if s["counterpart_id"] is not None]
                if not english_sentences:
                    return

                english_samples = random.sample(english_sentences, min(n, len(english_sentences)))

                # Get their counterparts
                carroll_samples = []
                for eng_sample in english_samples:
                    carroll_sample = generator.db.get_sentence_where(id=eng_sample["counterpart_id"])[0]
                    carroll_samples.append(carroll_sample)

                # Write section header
                header = f"### {sentence_type}_{subtype}" if subtype else f"### {sentence_type}"
                f.write(f"{header}\n\n")

                # Write samples
                for eng, car in zip(english_samples, carroll_samples):
                    f.write(f"English: {eng['sentence']} :: `{eng['form']}`\n")
                    f.write(f"Carroll: {car['sentence']} :: `{car['form']}`\n\n")

            # Domain constraint
            f.write("## Domain Constraint\n\n")
            write_section("domain_constraint")

            # Atomic sentences
            f.write("## Atomic Sentences\n\n")
            write_section("atomic", "monadic")
            write_section("atomic", "dyadic")

            # Simple negations
            f.write("## Simple Negations\n\n")
            write_section("negation", "monadic")
            write_section("negation", "dyadic")

            # Quantified sentences
            f.write("## Quantified Sentences\n\n")

            # Basic quantified forms
            write_section("quantified", "universal_affirmative")
            write_section("quantified", "particular_affirmative")
            write_section("quantified", "universal_negative")
            write_section("quantified", "particular_negative")
            write_section("quantified", "only_restrictor")

            # Name-based quantified forms
            write_section("quantified", "name_all")
            write_section("quantified", "name_some")

            # All-All variations
            write_section("quantified", "all_all")
            write_section("quantified", "all_all_all")
            write_section("quantified", "all_all_back")

            # All-Some variations
            write_section("quantified", "all_some")
            write_section("quantified", "all_some_back")

            # Some-All variations
            write_section("quantified", "some_all")
            write_section("quantified", "some_all_back")

            # Some-Some variations
            write_section("quantified", "some_some")
            write_section("quantified", "some_some_back")
            write_section("quantified", "some_some_some")

            # No variations
            write_section("quantified", "no_all")
            write_section("quantified", "no_some")
            write_section("quantified", "no_some_back")

            # Reverse variations
            write_section("quantified", "rev_some_all")
            write_section("quantified", "rev_no_all")

            # Self variations
            write_section("quantified", "some_self")

            # Connective sentences
            f.write("## Connective Sentences\n\n")

            # Conjunctions
            f.write("### Conjunctions\n\n")
            write_section("conjunction", "simple")
            write_section("conjunction", "vp_ellipsis")
            write_section("conjunction", "contrastive")

            # Disjunctions
            f.write("### Disjunctions\n\n")
            write_section("disjunction", "simple")
            write_section("disjunction", "vp_ellipsis")
            write_section("disjunction", "unless")
            write_section("disjunction", "negated_disjunct")

            # Conditionals
            f.write("### Conditionals\n\n")
            write_section("conditional", "if_then")
            write_section("conditional", "only_if")
            write_section("conditional", "nested")

            # Biconditionals
            f.write("### Biconditionals\n\n")
            write_section("biconditional", "if_and_only_if")
            write_section("biconditional", "just_in_case")

    # Generate all sentences
    generate_all_sentences()

    # Create samples
    # create_samples(n=4)
