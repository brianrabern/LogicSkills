import re
import json
import os
from typing import List, Set, Dict, Tuple
from Database.DB import db
from Database.models import Sentence
from Syntax.lexicon import Lexicon
from datetime import datetime
from Utils.normalize import normalize_logical_form, unescape_logical_form
from Semantics.eval import evaluate


def collect_all_predicates_and_names(forms: List[str]) -> Tuple[List[str], List[str]]:
    """Collect all unique predicates and names from a list of forms."""
    all_predicates: Set[str] = set()
    all_names: Set[str] = set()

    for form in forms:
        predicates = set(re.findall(r"([A-Z])", form))
        names = set(re.findall(r"([a-r])", form))
        all_predicates.update(predicates)
        all_names.update(names)

    return sorted(all_predicates), sorted(all_names)


def create_global_mapping(predicates: List[str], names: List[str]) -> Dict[str, str]:
    """Create a mapping for all predicates and names."""
    mapping = {}
    for i, pred in enumerate(predicates, 1):
        mapping[f"{{#{i}}}"] = pred
    for i, name in enumerate(names, 1):
        mapping[f"{{${i}}}"] = name
    return mapping


def abstract_form(form: str, mapping: Dict[str, str]) -> str:
    """Convert a logical form into an abstract pattern using the given mapping."""
    # Create reverse mapping for replacements
    reverse_mapping = {v: k for k, v in mapping.items()}
    print(f"\nAbstracting form: {form}")
    print(f"Reverse mapping: {reverse_mapping}")

    # Replace predicates and names in form
    abstracted = form
    for orig, placeholder in reverse_mapping.items():
        print(f"Replacing {orig} with {placeholder}")
        abstracted = abstracted.replace(orig, placeholder)
        print(f"After replacement: {abstracted}")

    print(f"Final abstracted form: {abstracted}")
    return abstracted


def get_alt_predicates(lexicon: Lexicon, original_predicate: str) -> List[str]:
    """Get valid predicates from lexicon that match the properties of the original predicate."""
    if original_predicate not in lexicon.predicates:
        return []

    original_info = lexicon.predicates[original_predicate]
    valid = []

    for symbol, info in lexicon.predicates.items():
        if (
            info["arity"] == original_info["arity"]
            and info["structure"] == original_info["structure"]
            and info["semantic_type"] == original_info["semantic_type"]
        ):
            valid.append(symbol)
    return valid


def generate_mappings(
    placeholders: List[str], alternatives: List[List[str]], used_symbols: Set[str] = None
) -> List[Dict[str, str]]:
    """Generate all possible mappings for placeholders to symbols."""
    if not placeholders:
        return [{}]

    if used_symbols is None:
        used_symbols = set()

    current_placeholder = placeholders[0]
    current_alternatives = [s for s in alternatives[0] if s not in used_symbols]

    if not current_alternatives:
        return []

    mappings = []
    for symbol in current_alternatives:
        new_used = used_symbols.copy()
        new_used.add(symbol)

        for sub_mapping in generate_mappings(placeholders[1:], alternatives[1:], new_used):
            mapping = {current_placeholder: symbol}
            mapping.update(sub_mapping)
            mappings.append(mapping)

    return mappings


def normalize_form(form: str) -> str:
    """Normalize a logical form by removing whitespace and standardizing Unicode representation."""
    # Remove all whitespace
    form = "".join(form.split())

    # Convert to ASCII representation of Unicode
    form = form.encode("unicode-escape").decode("ascii")

    return form


def find_matching_sentences(form: str, sentence_type: str = None) -> List[Sentence]:
    """Find all sentences in the database that match the given logical form, optionally filtered by type."""
    try:
        # First try exact match with database filtering
        query = db.session.query(Sentence)

        if sentence_type:
            query = query.filter(Sentence.type == sentence_type)

        # Try exact match first
        matching_forms = query.filter(Sentence.form == form).all()

        if matching_forms:
            return matching_forms

        # If no exact match, try normalized match but only for sentences of the right type
        normalized_form = normalize_logical_form(form)
        all_sentences = query.all()  # This is now filtered by type if provided

        matching_forms = []
        for sentence in all_sentences:
            if isinstance(sentence.form, str):
                try:
                    form_json = json.loads(sentence.form)
                    db_form = form_json[0] if isinstance(form_json, list) else form_json
                except json.JSONDecodeError:
                    db_form = sentence.form
            else:
                db_form = sentence.form

            # Normalize both forms to escaped Unicode format
            normalized_db_form = normalize_logical_form(db_form)
            if normalized_db_form == normalized_form:
                print(f"Found normalized match - ID: {sentence.id}, Form: {db_form}")
                matching_forms.append(sentence)

        if not matching_forms:
            print(f"No matches found for form: {form}")
            print(f"Normalized form: {normalized_form}")
        return matching_forms
    except Exception as e:
        print(f"Error finding matching sentences: {e}")
        return []


def find_matching_argument(instance, argument):
    """Find a complete matching argument in the database for the given instance."""
    matching_sentences = []

    for form, original_sentence in zip(instance, argument.values()):
        print(f"\nLooking for matches for form: {form}")
        matches = find_matching_sentences(form, sentence_type=original_sentence["type"])

        if not matches:
            print(f"No matches found for form: {form}")
            return None

        # Take the first match since we're looking for exact matches of our generated instances
        matching_sentences.append(matches[0])
        print(f"Using match with ID: {matches[0].id}")

    return matching_sentences


def generate_argument_instances(abstract_patterns: List[str], global_mapping: Dict[str, str]) -> List[List[str]]:
    """Generate all possible argument instances."""
    lexicon = Lexicon()
    instances = []

    # 1. Get alternatives for each predicate placeholder
    predicate_alternatives = {}
    for placeholder, orig_pred in global_mapping.items():
        if placeholder.startswith("{#"):
            predicate_alternatives[placeholder] = get_alt_predicates(lexicon, orig_pred)

    print("\nPredicate alternatives:")
    for placeholder, alternatives in predicate_alternatives.items():
        print(f"  {placeholder} ({global_mapping[placeholder]}): {alternatives}")

    # 2. Get name placeholders
    name_placeholders = set()
    for pattern in abstract_patterns:
        name_placeholders.update(re.findall(r"\{\$\d+\}", pattern))
    name_placeholders = sorted(list(name_placeholders))
    valid_names = list(lexicon.names.keys())

    print(f"\nName placeholders: {name_placeholders}")
    print(f"Valid names: {valid_names}")

    # 3. Generate all possible combinations
    all_placeholders = sorted(predicate_alternatives.keys()) + sorted(name_placeholders)
    all_alternatives = [
        predicate_alternatives[p] if p in predicate_alternatives else valid_names for p in all_placeholders
    ]
    valid_mappings = generate_mappings(all_placeholders, all_alternatives)

    # Generate instances
    for mapping in valid_mappings:
        print(f"\nGenerating instance with mapping: {mapping}")
        concrete_forms = []
        for pattern in abstract_patterns:
            concrete = pattern
            for placeholder, value in mapping.items():
                print(f"Replacing {placeholder} with {value}")
                concrete = concrete.replace(placeholder, value)
                print(f"After replacement: {concrete}")
            concrete_forms.append(concrete)
        instances.append(concrete_forms)

    return instances


if __name__ == "__main__":
    try:
        # Create output directory if it doesn't exist
        output_dir = "match_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create timestamped output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"matches_{timestamp}.json")
        # ∃x(Mx∧∀y(Ny→Ryx)),∀x(Mx→∀y((Ny∧∀z(Mz→Ryz))→Rxy)), (Nc∧∀y(My→Rcy)) |= ∃x(Mx∧Rxc)
        argument = {
            "premise1": {"form": "∃x(Mx∧∀y(Ny→Ryx))", "type": "quantified"},
            "premise2": {"form": "∀x(Mx→∀y((Ny∧∀z(Mz→Ryz))→Rxy))", "type": "quantified"},
            "premise3": {"form": "(Nc∧∀x(Mx→Rcx))", "type": "conjunction"},
            "conclusion": {"form": "∃x(Mx∧Rxc)", "type": "quantified"},
        }

        # Define the argument pattern to search for
        argument_pattern = [normalize_logical_form(form["form"]) for form in argument.values()]

        print("\nProcessing custom argument pattern")
        print("Premises:")
        for i, premise in enumerate(argument_pattern[:-1]):
            print(f"  {i+1}: {premise}")
        print(f"Conclusion: {argument_pattern[-1]}")

        predicates, names = collect_all_predicates_and_names(argument_pattern)
        global_mapping = create_global_mapping(predicates, names)

        print("\nPredicates found:", predicates)
        print("Names found:", names)
        print("\nGlobal mapping:", global_mapping)

        print("\nAbstracted patterns:")
        abstract_patterns = []
        for form in argument_pattern:
            abstracted = abstract_form(form, global_mapping)
            abstract_patterns.append(abstracted)
            print(f"Original: {form}")
            print(f"Abstracted: {abstracted}")
            print("---")

        # 3. Generate and count instances
        print("\nGenerating argument instances...")
        instances = generate_argument_instances(abstract_patterns, global_mapping)
        print(f"\nGenerated {len(instances)} total instances")

        # 4. Try to find matches
        print("\nSearching for matches...")
        matches_found = 0
        matches_found_list = []
        for i, instance in enumerate(instances, 1):
            if i % 100 == 0:
                print(f"Checked {i} instances...")

            matching_sentences = find_matching_argument(instance, argument)
            if matching_sentences:
                matches_found += 1
                matches_found_list.append(matching_sentences)
                print(f"\nFound complete argument match #{matches_found}:")
                for j, sentence in enumerate(matching_sentences, 1):
                    print(f"Sentence {j}: {sentence.form} (ID: {sentence.id})")

        print(f"\nTotal matches found: {matches_found}")
        print(f"Total instances checked: {len(instances)}")
        print("Matches found:")
        for match in matches_found_list:
            premises = match[:-1]
            conclusion = match[-1]
            print("Premises:")
            for i, premise in enumerate(premises):
                print(f"  {i+1}: {unescape_logical_form(premise.form)} (ID: {premise.id})")
            print(f"Conclusion: {unescape_logical_form(conclusion.form)} (ID: {conclusion.id})")

            # Validate and save the argument
            from Generators.arg_generator import ArgGenerator

            generator = ArgGenerator(db.session, evaluate)
            saved_argument = generator.validate_and_save_argument(premises, conclusion)
            if saved_argument:
                print(f"Successfully saved argument with ID: {saved_argument.id}")
            else:
                print("Failed to save argument")

    except Exception as e:
        print(f"Error during execution: {e}")
