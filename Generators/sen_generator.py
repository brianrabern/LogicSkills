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
	def __init__(self, lexicon, db=db):
		self.lexicon = lexicon
		self.counter = 0
		self.entries = []
		self.db = db
		self.language = lexicon.language if lexicon.language is not None else None

	def _sentence_exists(self, form, type=None, subtype=None):
		"""Check if a sentence with the given form already exists in the database."""
		try:
			normalized_form = normalize_logical_form(form)
			return self.db.sentence_exists(normalized_form, type, subtype)
		except Exception as e:
			logger.error(f"Error checking for existing sentence: {e}")
			return False

	def add_entry(self, sentence, type, subtype, soa, form, base=False):
		timestamp = int(time.time())
		# Normalize the form to escaped Unicode format
		raw_ast = self._parse_ast(form)
		normalized_form = normalize_logical_form(form)

		# Check if the sentence already exists
		if self._sentence_exists(normalized_form, type, subtype):
			logger.info(f"Sentence already exists: {sentence}")
			return

		status = self._get_status(raw_ast)
		try:
			self.db.add_sentence(
				sentence=sentence,
				type=type,
				subtype=subtype,
				soa=soa,
				form=normalized_form,  # Store the normalized form
				ast=self._ast_to_json_compatible(raw_ast),
				base=1 if base else 0,
				status=status,
				time_created=timestamp,
				language=self.language,
			)
			self.counter += 1
			logger.info(f"{self.counter}: {sentence}")

		except Exception as e:
			logger.error(f"Error adding entry: {e}")

	def _parse_ast(self, form):
		logger.info(f"Parsing AST for: {form}")
		try:
			tree = parser.parse(form)
			ast = transformer.transform(tree)
		except Exception as e:
			logger.error(f"Parse error: {str(e)}")
			ast = None
		return ast

	def _get_status(self, raw_ast):
		result = evaluate(raw_ast)
		if result == "unsat":
			return -1  # logical falsehood
		elif evaluate(("not", raw_ast)) == "unsat":
			return 1  # theorem
		elif result == "sat":
			return 0  # contingent
		else:
			return None  # evaluation failed or unknown

	def get_entries(self):
		try:
			return self.db.get_all_sentences()
		except Exception as e:
			logger.error(f"Error retrieving entries: {e}")

	def get_base_entries(self):
		try:
			return self.db.get_base_sentences()
		except Exception as e:
			logger.error(f"Error retrieving base entries: {e}")

	def generate_domain_constraint(self):
		kind_predicates = {
			k: v for k, v in self.lexicon.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
		}
		or_clause = ""
		soa = {}
		for kind_symbol, kind_data in kind_predicates.items():
			soa[kind_symbol] = kind_data["template"]
			kind_noun = kind_data["template"].replace("[1] is a ", "").strip()
			art = "an" if kind_noun[0].lower() in "aeiou" else "a"
			or_clause += f"{art} {kind_noun}, or "

		or_clause = or_clause.rstrip(", or ")
		sentence = f"Everything is {or_clause} (exclusively), and there's at least one of each."
		kinds = list(kind_predicates.keys())
		logger.info(sentence)

		def construct_universal_clause(kinds):
			# Start with the last kind
			result = f"{kinds[-1]}x"

			# Add the rest in reverse order with proper nesting
			for kind in reversed(kinds[:-1]):
				result = f"({kind}x∨{result})"

			return f"∀x{result}"

		def construct_exclusivity_clauses(kinds):
			clauses = []
			for i in range(len(kinds)):
				for j in range(i + 1, len(kinds)):
					clause = f"¬∃x({kinds[i]}x∧{kinds[j]}x)"
					clauses.append(clause)
			# Start with the last clause
			result = clauses[-1]
			# Add the rest in reverse order with proper nesting
			for clause in reversed(clauses[:-1]):
				result = f"({clause} ∧ {result})"

			return result

		def construct_existence_clauses(kinds):
			# Start with the last kind
			result = f"∃x{kinds[-1]}x"

			# Add the rest in reverse order with proper nesting
			for kind in reversed(kinds[:-1]):
				result = f"(∃x{kind}x ∧ {result})"

			return result

		conjunct1 = construct_universal_clause(kinds)
		conjunct2 = construct_exclusivity_clauses(kinds)
		conjunct3 = construct_existence_clauses(kinds)

		entry = {
			"sentence": sentence,
			"type": "domain_constraint",
			"subtype": None,
			"soa": soa,
			"form": f"(({conjunct1} ∧ {conjunct2}) ∧ {conjunct3})",
		}
		logger.info(entry)

		self.add_entry(**entry)

	def generate_atomic_sentences(self, vp_ellipsis=True):
		for pred_symbol, pred in self.lexicon.predicates.items():
			arity = pred["arity"]
			template = pred["template"]
			structure = pred["structure"]
			semantic_type = pred["semantic_type"]
			neg_template = pred["negated_template"]
			names = self.lexicon.names

			if arity == 1:
				for name_symbol in self.lexicon.names:
					name = self.lexicon.get_name(name_symbol)

					form = f"{pred_symbol}{name_symbol}"
					soa = {pred_symbol: template, name_symbol: name}
					sentence = template.replace("[1]", name) + "."
					self.add_entry(
						sentence=sentence,
						type="atomic",
						subtype="monadic",
						soa=soa,
						form=form,
						base=True,
					)

					# vp ellipsis cases
					if vp_ellipsis:
						for name_symbol2 in names:
							if semantic_type == "kind":
								continue
							if name_symbol != name_symbol2:
								name2 = self.lexicon.get_name(name_symbol2)
								expanded_soa = {
									pred_symbol: template,
									name_symbol: name,
									name_symbol2: name2,
								}
								if structure.startswith("copula"):
									disjunction_vpe = template.replace("[1] is", f"{name} or {name2} are") + "."
								else:
									disjunction_vpe = template.replace("[1]", f"{name} or {name2}") + "."
								disjunction_form = f"({pred_symbol}{name_symbol}∨{pred_symbol}{name_symbol2})"
								self.add_entry(
									disjunction_vpe,
									"disjunction",
									"vp_ellipsis",
									expanded_soa,
									disjunction_form,
								)
								if structure.startswith("copula"):
									conjunction_vpe = template.replace("[1] is", f"{name} and {name2} are") + "."
								else:
									conjunction_vpe = template.replace("[1]", f"{name} and {name2}") + "."
								conjunction_form = f"({pred_symbol}{name_symbol}∧{pred_symbol}{name_symbol2})"
								self.add_entry(
									conjunction_vpe,
									"conjunction",
									"vp_ellipsis",
									expanded_soa,
									conjunction_form,
								)
					neg_form = f"\u00ac{pred_symbol}{name_symbol}"
					neg_sentence = neg_template.replace("[1]", name) + "."
					self.add_entry(neg_sentence, "negation", "monadic", soa, neg_form, base=True)
			elif arity == 2:
				for name1 in self.lexicon.names:
					for name2 in self.lexicon.names:
						name_1 = self.lexicon.get_name(name1)
						name_2 = self.lexicon.get_name(name2)

						form = f"{pred_symbol}{name1}{name2}"
						soa = {pred_symbol: template, name1: name_1, name2: name_2}
						sentence = template.replace("[1]", name_1).replace("[2]", name_2) + "."
						self.add_entry(sentence, "atomic", "dyadic", soa, form, base=True)
						neg_form = f"\u00ac{pred_symbol}{name1}{name2}"
						neg_sentence = neg_template.replace("[1]", name_1).replace("[2]", name_2) + "."
						self.add_entry(
							neg_sentence,
							"negation",
							"monadic",
							soa,
							neg_form,
							base=True,
						)

	def generate_simple_quantified_sentences(self):
		kind_predicates = {
			k: v for k, v in self.lexicon.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
		}

		for pred_symbol, pred_data in self.lexicon.predicates.items():
			if pred_symbol in kind_predicates:
				continue  # skip restrictors as main predicates

			arity = pred_data["arity"]
			pred_template = pred_data["template"]

			for restrictor_symbol, restrictor_data in kind_predicates.items():
				restrictor_label = restrictor_data["template"]
				kind_noun = restrictor_label.replace("[1] is a ", "").strip()
				art = "An" if kind_noun[0].lower() in "aeiou" else "A"

				if arity == 1:
					base = pred_template.replace("[1]", "").strip()
					copula_plural = (
						base.replace(" is ", " are ") if pred_data["structure"].startswith("copula") else base
					)
					universal_affirmative = {
						"sentence": f"Every {kind_noun} {base}.",
						"type": "quantified",
						"subtype": "universal_affirmative",
						"soa": {
							restrictor_symbol: restrictor_label,
							pred_symbol: pred_template,
						},
						"form": f"∀x({restrictor_symbol}x→{pred_symbol}x)",
						"base": True,
					}
					self.add_entry(**universal_affirmative)

					particular_affirmative = {
						"sentence": f"{art} {kind_noun} {base}.",
						"type": "quantified",
						"subtype": "particular_affirmative",
						"soa": {
							restrictor_symbol: restrictor_label,
							pred_symbol: pred_template,
						},
						"form": f"∃x({restrictor_symbol}x∧{pred_symbol}x)",
						"base": True,
					}
					self.add_entry(**particular_affirmative)

					universal_negative = {
						"sentence": f"No {kind_noun}s {copula_plural}.",
						"type": "quantified",
						"subtype": "universal_negative",
						"soa": {
							restrictor_symbol: restrictor_label,
							pred_symbol: pred_template,
						},
						"form": f"¬∃x({restrictor_symbol}x∧{pred_symbol}x)",
						"base": True,
					}
					self.add_entry(**universal_negative)

					particular_negative = {
						"sentence": f"Not all {kind_noun}s {copula_plural}.",
						"type": "quantified",
						"subtype": "particular_negative",
						"soa": {
							restrictor_symbol: restrictor_label,
							pred_symbol: pred_template,
						},
						"form": f"∃x({restrictor_symbol}x∧¬{pred_symbol}x)",
						"base": True,
					}
					self.add_entry(**particular_negative)

					# Only A's are B :: ∀x(Bx → Ax)
					only_restrictor = {
						"sentence": f"Only {kind_noun}s {copula_plural}.",
						"type": "quantified",
						"subtype": "only_restrictor",
						"soa": {
							restrictor_symbol: restrictor_label,
							pred_symbol: pred_template,
						},
						"form": f"∀x({pred_symbol}x→{restrictor_symbol}x)",
						"base": True,
					}
					self.add_entry(**only_restrictor)

				elif arity == 2:
					for name_sym, name_data in self.lexicon.names.items():
						name = name_data["name"]

						base = pred_template.replace("[1]", "").replace("[2]", name).strip()

						universal_affirmative = {
							"sentence": f"Every {kind_noun} {base}.",
							"type": "quantified",
							"subtype": "universal_affirmative",
							"soa": {
								restrictor_symbol: restrictor_label,
								pred_symbol: pred_template,
								name_sym: name,
							},
							"form": f"∀x({restrictor_symbol}x→{pred_symbol}x{name_sym})",
							"base": True,
						}
						self.add_entry(**universal_affirmative)
						particular_affirmative = {
							"sentence": f"{art} {kind_noun} {base}.",
							"type": "quantified",
							"subtype": "particular_affirmative",
							"soa": {
								restrictor_symbol: restrictor_label,
								pred_symbol: pred_template,
								name_sym: name,
							},
							"form": f"∃x({restrictor_symbol}x∧{pred_symbol}x{name_sym})",
							"base": True,
						}
						self.add_entry(**particular_affirmative)
						universal_negative = {
							"sentence": f"No {kind_noun}s {base}.",
							"type": "quantified",
							"subtype": "universal_negative",
							"soa": {
								restrictor_symbol: restrictor_label,
								pred_symbol: pred_template,
								name_sym: name,
							},
							"form": f"¬∃x({restrictor_symbol}x∧{pred_symbol}x{name_sym})",
							"base": True,
						}
						self.add_entry(**universal_negative)

						particular_negative = {
							"sentence": f"Not all {kind_noun}s {base}.",
							"type": "quantified",
							"subtype": "particular_negative",
							"soa": {
								restrictor_symbol: restrictor_label,
								pred_symbol: pred_template,
								name_sym: name,
							},
							"form": f"∃x({restrictor_symbol}x∧¬{pred_symbol}x{name_sym})",
							"base": True,
						}

						self.add_entry(**particular_negative)

	def generate_complex_dyadic_sentences(self):
		# Generate sentences with three quantifiers and dyadic predicates
		kind_predicates = {
			k: v for k, v in self.lexicon.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
		}
		dyadic_predicates = {}
		for pred_symbol, pred_data in self.lexicon.predicates.items():
			if pred_data["arity"] == 2:
				dyadic_predicates[pred_symbol] = pred_data

		for pred_symbol, pred_data in dyadic_predicates.items():
			if pred_symbol in kind_predicates:
				continue
			relation = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()

			for restrictor_symbol1, restrictor_data1 in kind_predicates.items():
				restrictor_label1 = restrictor_data1["template"]
				kind_noun1 = restrictor_label1.replace("[1] is a ", "").strip()
				art1 = "An" if kind_noun1[0].lower() in "aeiou" else "A"

				for restrictor_symbol2, restrictor_data2 in kind_predicates.items():
					restrictor_label2 = restrictor_data2["template"]
					kind_noun2 = restrictor_label2.replace("[1] is a ", "").strip()
					art2 = "An" if kind_noun2[0].lower() in "aeiou" else "A"

					for restrictor_symbol3, restrictor_data3 in kind_predicates.items():
						restrictor_label3 = restrictor_data3["template"]
						kind_noun3 = restrictor_label3.replace("[1] is a ", "").strip()
						art3 = "An" if kind_noun3[0].lower() in "aeiou" else "A"

						soa = {
							restrictor_symbol1: restrictor_label1,
							restrictor_symbol2: restrictor_label2,
							restrictor_symbol3: restrictor_label3,
							pred_symbol: pred_data["template"],
						}

						# all_all_all: Every A Rs every B that Rs every C :: ∀x(Ax→∀y((By∧∀z(Cz→Ryz))→Rxy))
						all_all_all = {
							"sentence": (
								f"Every {kind_noun1} {relation} every {kind_noun2} that {relation} every {kind_noun3}."
							),
							"type": "quantified",
							"subtype": "all_all_all",
							"soa": soa,
							"form": f"∀x({restrictor_symbol1}x→∀y(({restrictor_symbol2}y∧∀z({restrictor_symbol3}z→{pred_symbol}yz))→{pred_symbol}xy))",
						}
						self.add_entry(**all_all_all)

						# some_some_some: Some A Rs some B that Rs some C :: ∃x∃y∃z(((Ax∧By)∧Cz)∧(Rxy∧Ryz))
						some_some_some = {
							"sentence": (
								f"{art1} {kind_noun1} {relation} {art2.lower()} {kind_noun2} that {relation} {art3.lower()} {kind_noun3}."
							),
							"type": "quantified",
							"subtype": "some_some_some",
							"soa": soa,
							"form": f"∃x∃y∃z((({restrictor_symbol1}x∧{restrictor_symbol2}y)∧{restrictor_symbol3}z)∧({pred_symbol}xy∧{pred_symbol}yz))",
						}
						self.add_entry(**some_some_some)

	def generate_multiply_quantified_sentences(self):
		# Generate sentences with multiple quantifiers
		kind_predicates = {
			k: v for k, v in self.lexicon.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
		}
		dyadic_predicates = {}
		for pred_symbol, pred_data in self.lexicon.predicates.items():
			if pred_data["arity"] == 2:
				dyadic_predicates[pred_symbol] = pred_data

		for pred_symbol, pred_data in dyadic_predicates.items():
			if pred_symbol in kind_predicates:
				continue
			relation = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
			for restrictor_symbol1, restrictor_data1 in kind_predicates.items():
				restrictor_label1 = restrictor_data1["template"]
				kind_noun1 = restrictor_label1.replace("[1] is a ", "").strip()
				art1 = "An" if kind_noun1[0].lower() in "aeiou" else "A"

				for restrictor_symbol2, restrictor_data2 in kind_predicates.items():
					restrictor_label2 = restrictor_data2["template"]
					kind_noun2 = restrictor_label2.replace("[1] is a ", "").strip()
					art2 = "An" if kind_noun2[0].lower() in "aeiou" else "A"

					soa = {
						restrictor_symbol1: restrictor_label1,
						restrictor_symbol2: restrictor_label2,
						pred_symbol: pred_data["template"],
					}

					# all_all:    Every A R's every B     :: ∀x∀y((Ax ∧ By) → Rxy)
					all_all = {
						"sentence": f"Every {kind_noun1} {relation} every {kind_noun2}.",
						"type": "quantified",
						"subtype": "all_all",
						"soa": soa,
						"form": f"∀x∀y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) → {pred_symbol}xy)",
					}
					self.add_entry(**all_all)
					# all_some:   Every A R's some B      :: ∀x(Ax → ∃y(By ∧ Rxy))
					all_some = {
						"sentence": f"Every {kind_noun1} {relation} {art2.lower()} {kind_noun2}.",
						"type": "quantified",
						"subtype": "all_some",
						"soa": soa,
						"form": f"∀x({restrictor_symbol1}x → ∃y({restrictor_symbol2}y ∧ {pred_symbol}xy))",
					}
					self.add_entry(**all_some)
					# some_all:   An A R's every B        :: ∃x(Ax ∧ ∀y(By → Rxy))
					some_all = {
						"sentence": f"{art1} {kind_noun1} {relation} every {kind_noun2}.",
						"type": "quantified",
						"subtype": "some_all",
						"soa": soa,
						"form": f"∃x({restrictor_symbol1}x ∧ ∀y({restrictor_symbol2}y → {pred_symbol}xy))",
					}
					self.add_entry(**some_all)
					# some_some:  An A R's a B            :: ∃x∃y((Ax ∧ By) ∧ Rxy)
					some_some = {
						"sentence": f"{art1} {kind_noun1} {relation} {art2.lower()} {kind_noun2}.",
						"type": "quantified",
						"subtype": "some_some",
						"soa": soa,
						"form": f"∃x∃y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) ∧ {pred_symbol}xy)",
					}
					self.add_entry(**some_some)
					# no_all:     No A R's every B        :: ¬∃x(Ax ∧ ∀y(By → Rxy))
					no_all = {
						"sentence": f"No {kind_noun1}s {relation} every {kind_noun2}.",
						"type": "quantified",
						"subtype": "no_all",
						"soa": soa,
						"form": f"¬∃x({restrictor_symbol1}x ∧ ∀y({restrictor_symbol2}y → {pred_symbol}xy))",
					}
					self.add_entry(**no_all)
					# no_some:    No A R's a B            :: ¬∃x∃y((Ax ∧ By) ∧ Rxy)
					no_some = {
						"sentence": f"No {kind_noun1}s {relation} {art2.lower()} {kind_noun2}.",
						"type": "quantified",
						"subtype": "no_some",
						"soa": soa,
						"form": f"¬∃x∃y(({restrictor_symbol1}x ∧ {restrictor_symbol2}y) ∧ {pred_symbol}xy)",
					}
					self.add_entry(**no_some)

					# rev_some_all: There is an A that every B Rs  :: ∃x(Ax∧∀y(By→Ryx))
					rev_some_all = {
						"sentence": f"There is {art1.lower()} {kind_noun1} that every {kind_noun2} {relation}.",
						"type": "quantified",
						"subtype": "rev_some_all",
						"soa": soa,
						"form": f"∃x({restrictor_symbol1}x∧∀y({restrictor_symbol2}y→{pred_symbol}yx))",
					}
					self.add_entry(**rev_some_all)

					# rev_no_all: There is not an A that every B Rs  :: ¬∃x(Ax∧∀y(By→Ryx))
					rev_no_all = {
						"sentence": f"There is not {art1.lower()} {kind_noun1} that every {kind_noun2} {relation}.",
						"type": "quantified",
						"subtype": "rev_no_all",
						"soa": soa,
						"form": f"¬∃x({restrictor_symbol1}x∧∀y({restrictor_symbol2}y→{pred_symbol}yx))",
					}
					self.add_entry(**rev_no_all)

				# some_self: An A Rs itself  :: ∃x(Ax ∧ Rxx)
				some_self = {
					"sentence": f"{art1} {kind_noun1} {relation} itself.",
					"type": "quantified",
					"subtype": "some_self",
					"soa": {
						restrictor_symbol1: restrictor_label1,
						pred_symbol: pred_data["template"],
					},
					"form": f"∃x({restrictor_symbol1}x ∧ {pred_symbol}xx)",
				}
				self.add_entry(**some_self)

	def generate_name_quantified_sentences(self):
		kind_predicates = {
			k: v for k, v in self.lexicon.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
		}
		dyadic_predicates = {}
		for pred_symbol, pred_data in self.lexicon.predicates.items():
			if pred_data["arity"] == 2:
				dyadic_predicates[pred_symbol] = pred_data

		for pred_symbol, pred_data in dyadic_predicates.items():
			if pred_symbol in kind_predicates:
				continue
			relation = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
			for restrictor_symbol, restrictor_data in kind_predicates.items():
				restrictor_label = restrictor_data["template"]
				kind_noun = restrictor_label.replace("[1] is a ", "").strip()
				art = "An" if kind_noun[0].lower() in "aeiou" else "A"
				for name_symbol in self.lexicon.names:
					name = self.lexicon.get_name(name_symbol)
					soa = {
						restrictor_symbol: restrictor_label,
						pred_symbol: pred_data["template"],
						name_symbol: name,
					}
					sentence = f"{name} {relation} every {kind_noun}."
					entry = {
						"sentence": sentence,
						"type": "quantified",
						"subtype": "name_all",
						"soa": soa,
						"form": f"∀x({restrictor_symbol}x→{pred_symbol}{name_symbol}x)",
					}
					self.add_entry(**entry)
					sentence = f"{name} {relation} {art.lower()} {kind_noun}."
					entry = {
						"sentence": sentence,
						"type": "quantified",
						"subtype": "name_some",
						"soa": soa,
						"form": f"∃x({restrictor_symbol}x∧{pred_symbol}{name_symbol}x)",
					}
					self.add_entry(**entry)

	def generate_reciprocal_quantified_sentences(self):
		# Generate sentences with reciprocal relationships
		kind_predicates = {
			k: v for k, v in self.lexicon.predicates.items() if v["semantic_type"] == "kind" and v["arity"] == 1
		}
		dyadic_predicates = {}
		for pred_symbol, pred_data in self.lexicon.predicates.items():
			if pred_data["arity"] == 2:
				dyadic_predicates[pred_symbol] = pred_data

		for pred_symbol, pred_data in dyadic_predicates.items():
			if pred_symbol in kind_predicates:
				continue
			relation = pred_data["template"].replace("[1]", "").replace("[2]", "").strip()
			for restrictor_symbol1, restrictor_data1 in kind_predicates.items():
				restrictor_label1 = restrictor_data1["template"]
				kind_noun1 = restrictor_label1.replace("[1] is a ", "").strip()
				art1 = "An" if kind_noun1[0].lower() in "aeiou" else "A"
				for restrictor_symbol2, restrictor_data2 in kind_predicates.items():
					restrictor_label2 = restrictor_data2["template"]
					kind_noun2 = restrictor_label2.replace("[1] is a ", "").strip()
					art2 = "An" if kind_noun2[0].lower() in "aeiou" else "A"
					soa = {
						restrictor_symbol1: restrictor_label1,
						restrictor_symbol2: restrictor_label2,
						pred_symbol: pred_data["template"],
					}

					# all_all_back: Every A Rs every B that Rs it :: ∀x(Ax→∀y((By∧Ryx)→Rxy))
					all_all_back = {
						"sentence": f"Every {kind_noun1} {relation} every {kind_noun2} that {relation} it.",
						"type": "quantified",
						"subtype": "all_all_back",
						"soa": soa,
						"form": (
							f"∀x({restrictor_symbol1}x→∀y(({restrictor_symbol2}y∧{pred_symbol}yx)→{pred_symbol}xy))"
						),
					}
					self.add_entry(**all_all_back)

					# some_all_back: Some A Rs every B that Rs it :: ∃x(Ax∧∀y((By∧Ryx)→Rxy))
					some_all_back = {
						"sentence": f"{art1} {kind_noun1} {relation} every {kind_noun2} that {relation} it.",
						"type": "quantified",
						"subtype": "some_all_back",
						"soa": soa,
						"form": (
							f"∃x({restrictor_symbol1}x∧∀y(({restrictor_symbol2}y∧{pred_symbol}yx)→{pred_symbol}xy))"
						),
					}
					self.add_entry(**some_all_back)

					# some_some_back: Some A Rs some B that Rs it :: ∃x∃y((Ax∧By)∧(Rxy∧Ryx))
					some_some_back = {
						"sentence": f"{art1} {kind_noun1} {relation} {art2.lower()} {kind_noun2} that {relation} it.",
						"type": "quantified",
						"subtype": "some_some_back",
						"soa": soa,
						"form": (
							f"∃x∃y(({restrictor_symbol1}x∧{restrictor_symbol2}y)∧({pred_symbol}xy∧{pred_symbol}yx))"
						),
					}
					self.add_entry(**some_some_back)

					# all_some_back: Every A Rs some B that Rs it :: ∀x(Ax→∃y(By∧(Rxy∧Ryx)))
					all_some_back = {
						"sentence": f"Every {kind_noun1} {relation} {art2.lower()} {kind_noun2} that {relation} it.",
						"type": "quantified",
						"subtype": "all_some_back",
						"soa": soa,
						"form": (
							f"∀x({restrictor_symbol1}x→∃y({restrictor_symbol2}y∧({pred_symbol}xy∧{pred_symbol}yx)))"
						),
					}
					self.add_entry(**all_some_back)

					# no_some_back: No A Rs some B that Rs it :: ¬∃x∃y((Ax∧By)∧(Rxy∧Ryx))
					no_some_back = {
						"sentence": f"No {kind_noun1} {relation} {art2.lower()} {kind_noun2} that {relation} it.",
						"type": "quantified",
						"subtype": "no_some_back",
						"soa": soa,
						"form": (
							f"¬∃x∃y(({restrictor_symbol1}x∧{restrictor_symbol2}y)∧({pred_symbol}xy∧{pred_symbol}yx))"
						),
					}
					self.add_entry(**no_some_back)

	def generate_conjunctions(
		self,
		sentence_types=None,
		only_atomic=False,
	):
		entries = self.get_base_entries()
		if only_atomic:
			entries = [e for e in entries if e["type"] == "atomic"]

		if sentence_types:
			for t in sentence_types:
				type = t["type"]
				subtype = t["subtype"]
				logger.info(f"Generating conjunctions for {type} {subtype}")
				res = self.db.get_sentence_where(type=t["type"], subtype=t["subtype"])
				entries.extend(res)
		sample_size = min(50, max(20, int(len(entries) * 0.3)))
		sampled_entries = random.sample(entries, sample_size)
		logger.info(f"Sampled {len(sampled_entries)} entries")

		for e1, e2 in itertools.combinations(sampled_entries, 2):
			# Prioritize combinations that mix quantifier types
			if (
				e1["type"] == "quantified"
				and e2["type"] == "quantified"
				and e1["form"].startswith("∀") != e2["form"].startswith("∀")
			):
				self._add_and(e1, e2)
				continue

			# Include combinations with binary predicates
			if any("∃" in e["form"] and "∧" in e["form"] for e in [e1, e2]):
				self._add_and(e1, e2)
				continue

			# Include some non-quantified combinations
			if random.random() < 0.5:  # 50% chance to include non-quantified combinations
				self._add_and(e1, e2)

	def generate_disjunctions(self, sentence_types=None, only_atomic=False):
		entries = self.get_base_entries()
		if only_atomic:
			entries = [e for e in entries if e["type"] == "atomic"]
		if sentence_types:
			for t in sentence_types:
				res = self.db.get_sentence_where(type=t["type"], subtype=t["subtype"])
				entries.extend(res)
		sample_size = min(50, max(20, int(len(entries) * 0.3)))
		sampled_entries = random.sample(entries, sample_size)

		for e1, e2 in itertools.combinations(sampled_entries, 2):
			# Skip if both entries are of the same type

			# Prioritize combinations that create interesting logical structures
			if e1["type"] == "quantified" and e2["type"] == "quantified" and "→" in e1["form"] and "→" in e2["form"]:
				self._add_or(e1, e2)
				continue

			# Include combinations that mix universal and existential
			if (
				e1["type"] == "quantified"
				and e2["type"] == "quantified"
				and e1["form"].startswith("∀") != e2["form"].startswith("∀")
			):
				self._add_or(e1, e2)
				continue

			# Include some non-quantified combinations
			if random.random() < 0.5:  # 50% chance to include non-quantified combinations
				self._add_or(e1, e2)

	def generate_conditionals(self, sentence_types=None, only_atomic=False):
		entries = self.get_base_entries()
		if only_atomic:
			entries = [e for e in entries if e["type"] == "atomic"]
		if sentence_types:
			for t in sentence_types:
				res = self.db.get_sentence_where(type=t["type"], subtype=t["subtype"])
				entries.extend(res)
		sample_size = min(50, max(20, int(len(entries) * 0.3)))
		sampled_entries = random.sample(entries, sample_size)

		for e1, e2 in itertools.combinations(sampled_entries, 2):
			# don't use only_restrictor quantified sentences
			if e1["subtype"] == "only_restrictor" or e2["subtype"] == "only_restrictor":
				continue

			# Prioritize combinations that create nested conditionals
			if e1["type"] == "quantified" and "→" in e1["form"] and e2["type"] == "quantified" and "→" in e2["form"]:
				self._add_conditional(e1, e2)
				continue

			# Include combinations that mix universal and existential
			if (
				e1["type"] == "quantified"
				and e2["type"] == "quantified"
				and e1["form"].startswith("∀") != e2["form"].startswith("∀")
			):
				self._add_conditional(e1, e2)
				continue

			# Include some non-quantified combinations
			if random.random() < 0.4:  # 40% chance to include non-quantified combinations
				self._add_conditional(e1, e2)

	def generate_biconditionals(self, sentence_types=None, only_atomic=False):
		entries = self.get_base_entries()
		if only_atomic:
			entries = [e for e in entries if e["type"] == "atomic"]
		if sentence_types:
			for t in sentence_types:
				res = self.db.get_sentence_where(type=t["type"], subtype=t["subtype"])
				entries.extend(res)
		sample_size = min(50, max(20, int(len(entries) * 0.3)))
		sampled_entries = random.sample(entries, sample_size)

		for e1, e2 in itertools.combinations(sampled_entries, 2):
			# Skip if both entries are of the same type
			if e1["subtype"] == "only_restrictor" or e2["subtype"] == "only_restrictor":
				continue

			# Prioritize combinations that create interesting equivalences
			if (
				e1["type"] == "quantified"
				and e2["type"] == "quantified"
				and e1["form"].startswith("∀") != e2["form"].startswith("∀")
			):
				self._add_iff(e1, e2)
				continue

			# Include combinations with binary predicates
			if any("∃" in e["form"] and "∧" in e["form"] for e in [e1, e2]):
				self._add_iff(e1, e2)
				continue

			# Include some non-quantified combinations
			if random.random() < 0.3:  # 30% chance to include non-quantified combinations
				self._add_iff(e1, e2)

	def generate_nested_conditionals(self):
		base_entries = self.get_base_entries()
		sample_size = min(50, max(20, int(len(base_entries) * 0.3)))
		sampled_entries = random.sample(base_entries, sample_size)

		for e1, e2, e3 in itertools.combinations(sampled_entries, 3):
			# Skip if all entries are of the same type
			if e1["type"] == e2["type"] == e3["type"]:
				continue
			if (
				e1["subtype"] == "only_restrictor"
				or e2["subtype"] == "only_restrictor"
				or e3["subtype"] == "only_restrictor"
			):
				continue
			# Only create if at least one entry is quantified
			if any(e["type"] == "quantified" for e in [e1, e2, e3]):
				self._add_if_then_only_if(e1, e2, e3)
			elif random.random() < 0.3:
				self._add_if_then_only_if(e1, e2, e3)

	# etc...
	def _add_and(self, e1, e2):
		s1 = self._capitalize(self._lowercase_except_names(e1["sentence"].rstrip(".")))
		s2 = self._lowercase_except_names(e2["sentence"].rstrip("."))
		soa = {**e1["soa"], **e2["soa"]}
		self.add_entry(
			sentence=f"{s1} and {s2}.",
			type="conjunction",
			subtype="simple",
			soa=soa,
			form=f"({e1['form']}∧{e2['form']})",
		)
		if e2["type"] != "quantified":
			self.add_entry(
				sentence=f"{s1} but it is not the case that {s2}.",
				type="conjunction",
				subtype="contrastive",
				soa=soa,
				form=f"({e1['form']}∧¬{e2['form']})",
			)

	def _add_or(self, e1, e2):
		s1 = self._capitalize(self._lowercase_except_names(e1["sentence"].rstrip(".")))
		s2 = self._lowercase_except_names(e2["sentence"].rstrip("."))
		soa = {**e1["soa"], **e2["soa"]}

		self.add_entry(
			sentence=f"{s1} or {s2}.",
			type="disjunction",
			subtype="simple",
			soa=soa,
			form=f"({e1['form']}∨{e2['form']})",
		)
		self.add_entry(
			sentence=f"{s1} unless {s2}.",
			type="disjunction",
			subtype="unless",
			soa=soa,
			form=f"(¬{e2['form']}→{e1['form']})",
		)
		if e2["type"] != "quantified":
			self.add_entry(
				sentence=f"{s1} or it is not the case that {s2}.",
				type="disjunction",
				subtype="negated_disjunct",
				soa=soa,
				form=f"({e1['form']}∨¬{e2['form']})",
			)

	def _add_conditional(self, e1, e2):
		s1 = self._lowercase_except_names(e1["sentence"].rstrip("."))
		s2 = self._lowercase_except_names(e2["sentence"].rstrip("."))
		soa = {**e1["soa"], **e2["soa"]}

		self.add_entry(
			sentence=f"If {s1}, then {s2}.",
			type="conditional",
			subtype="if_then",
			soa=soa,
			form=f"({e1['form']}→{e2['form']})",
		)
		s1 = self._capitalize(s1)

		self.add_entry(
			sentence=f"{s1} only if {s2}.",
			type="conditional",
			subtype="only_if",
			soa=soa,
			form=f"({e1['form']}→{e2['form']})",
		)

	def _add_iff(self, e1, e2):
		s1 = self._capitalize(self._lowercase_except_names(e1["sentence"].rstrip(".")))
		s2 = self._lowercase_except_names(e2["sentence"].rstrip("."))
		soa = {**e1["soa"], **e2["soa"]}

		# randomly use "if and only if" or "just in case"
		flip_coin = random.choice([True, False])
		iff = "if and only if" if flip_coin else "just in case"
		self.add_entry(
			sentence=f"{s1} {iff} {s2}.",
			type="biconditional",
			subtype="if_and_only_if" if flip_coin else "just_in_case",
			soa=soa,
			form=f"({e1['form']}↔{e2['form']})",
		)

	def _add_if_then_only_if(self, e1, e2, e3):
		s1 = self._lowercase_except_names(e1["sentence"].rstrip("."))
		s2 = self._lowercase_except_names(e2["sentence"].rstrip("."))
		s3 = self._lowercase_except_names(e3["sentence"].rstrip("."))
		soa = {**e1["soa"], **e2["soa"], **e3["soa"]}
		self.add_entry(
			sentence=f"If {s1}, then {s2} only if {s3}.",
			type="conditional",
			subtype="nested",
			soa=soa,
			form=f"({e1['form']}→({e2['form']}→{e3['form']}))",
		)

	def _ast_to_json_compatible(self, raw_ast):
		if isinstance(raw_ast, tuple):
			return [self._ast_to_json_compatible(x) for x in raw_ast]
		return raw_ast

	def _lowercase_except_names(self, sentence):
		names = {data["name"] for data in self.lexicon.names.values()}
		words = sentence.split()
		if not words:
			return sentence
		return " ".join(word if word in names else word.lower() for word in words)

	def _capitalize(self, sentence):
		return sentence[0].upper() + sentence[1:] if sentence else sentence


if __name__ == "__main__":
	from Syntax.carroll_lexicon import CarrollLexicon

	lex = CarrollLexicon()
	generator = SentenceGenerator(lex)
	generator.generate_biconditionals()
	generator.generate_conditionals()
	generator.generate_nested_conditionals()
	generator.generate_conjunctions()


	def generate_all_sentences():
		"""Generate all sentence types without sampling."""
		lex = CarrollLexicon()
		generator = SentenceGenerator(lex)
		print("Generating domain constraint...")
		generator.generate_domain_constraint()

		print("Generating atomic sentences...")
		generator.generate_atomic_sentences()

		print("Generating simple quantified sentences...")
		generator.generate_simple_quantified_sentences()

		print("Generating multiply quantified sentences...")
		generator.generate_multiply_quantified_sentences()

		print("Generating reciprocal quantified sentences...")
		generator.generate_reciprocal_quantified_sentences()

		print("Generating complex dyadic sentences...")
		generator.generate_complex_dyadic_sentences()

		print("Generating name quantified sentences...")
		generator.generate_name_quantified_sentences()

		print("Generating compound sentences...")
		generator.generate_conjunctions()
		generator.generate_disjunctions()
		generator.generate_conditionals()
		generator.generate_biconditionals()
		generator.generate_nested_conditionals()

	# def create_samples(n=5):
	# 	"""Create a markdown file with samples of each sentence type."""
	# 	with open("sentence_samples.md", "w") as f:
	# 		f.write("# ArgBench Sentence Samples\n\n")
	# 		f.write(
	# 			"This file contains samples of all sentence types generated by the ArgBench sentence generator.\n\n"
	# 		)

	# 		def write_section(sentence_type, subtype=None):
	# 			kwargs = {"type": sentence_type}
	# 			if subtype:
	# 				kwargs["subtype"] = subtype
	# 			sentences = db.get_sentence_where(**kwargs)
	# 			samples = random.sample(sentences, min(n, len(sentences)))

	# 			# Write section header
	# 			header = f"### {sentence_type}_{subtype}" if subtype else f"### {sentence_type}"
	# 			f.write(f"{header}\n\n")

	# 			# Write samples
	# 			for s in samples:
	# 				f.write(f"- {s['sentence']} :: `{s['form']}`\n")
	# 			f.write("\n")

	# 		# Domain constraint
	# 		f.write("## Domain Constraint\n\n")
	# 		write_section("domain_constraint")

	# 		# Atomic sentences
	# 		f.write("## Atomic Sentences\n\n")
	# 		write_section("atomic", "monadic")
	# 		write_section("atomic", "dyadic")

	# 		# Simple negations
	# 		f.write("## Simple Negations\n\n")
	# 		write_section("negation", "monadic")
	# 		write_section("negation", "dyadic")

	# 		# Quantified sentences
	# 		f.write("## Quantified Sentences\n\n")

	# 		# Basic quantified forms
	# 		write_section("quantified", "universal_affirmative")
	# 		write_section("quantified", "particular_affirmative")
	# 		write_section("quantified", "universal_negative")
	# 		write_section("quantified", "particular_negative")
	# 		write_section("quantified", "only_restrictor")

	# 		# Name-based quantified forms
	# 		write_section("quantified", "name_all")
	# 		write_section("quantified", "name_some")

	# 		# All-All variations
	# 		write_section("quantified", "all_all")
	# 		write_section("quantified", "all_all_all")
	# 		write_section("quantified", "all_all_back")

	# 		# All-Some variations
	# 		write_section("quantified", "all_some")
	# 		write_section("quantified", "all_some_back")

	# 		# Some-All variations
	# 		write_section("quantified", "some_all")
	# 		write_section("quantified", "some_all_back")

	# 		# Some-Some variations
	# 		write_section("quantified", "some_some")
	# 		write_section("quantified", "some_some_back")
	# 		write_section("quantified", "some_some_some")

	# 		# No variations
	# 		write_section("quantified", "no_all")
	# 		write_section("quantified", "no_some")
	# 		write_section("quantified", "no_some_back")

	# 		# Reverse variations
	# 		write_section("quantified", "rev_some_all")
	# 		write_section("quantified", "rev_no_all")

	# 		# Self variations
	# 		write_section("quantified", "some_self")

	# 		# Connective sentences
	# 		f.write("## Connective Sentences\n\n")

	# 		# Conjunctions
	# 		f.write("### Conjunctions\n\n")
	# 		write_section("conjunction", "simple")
	# 		write_section("conjunction", "vp_ellipsis")
	# 		write_section("conjunction", "contrastive")

	# 		# Disjunctions
	# 		f.write("### Disjunctions\n\n")
	# 		write_section("disjunction", "simple")
	# 		write_section("disjunction", "vp_ellipsis")
	# 		write_section("disjunction", "unless")
	# 		write_section("disjunction", "negated_disjunct")

	# 		# Conditionals
	# 		f.write("### Conditionals\n\n")
	# 		write_section("conditional", "if_then")
	# 		write_section("conditional", "only_if")
	# 		write_section("conditional", "nested")

	# 		# Biconditionals
	# 		f.write("### Biconditionals\n\n")
	# 		write_section("biconditional", "if_and_only_if")
	# 		write_section("biconditional", "just_in_case")

	def generate():
		"""Generate all sentences."""
		# First generate all sentences
		generate_all_sentences()

		# # Then create the markdown file with samples
		# print("\nCreating samples markdown file...")
		# create_samples()

	# generate()
