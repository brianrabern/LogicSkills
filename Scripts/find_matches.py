import re
import json
import os
import sys
from typing import List, Set, Dict, Tuple
from Database.DB import db
from Database.models import Sentence, Argument
from Utils.normalize import normalize_logical_form

# from Semantics.eval import evaluate


def grab_argument_data(session, argument_id: str):
    """Grab an argument by ID."""
    # Get the argument
    argument = session.query(Argument).filter_by(id=argument_id).first()
    if not argument:
        print(f"Argument {argument_id} not found")
        return None
    # Get the premises
    premise_ids = argument.premise_ids.split(",")
    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()

    # Get the conclusion
    conclusion = session.query(Sentence).filter_by(id=argument.conclusion_id).first()

    argument_data = {
        "premise1": {"form": premises[0].form, "type": premises[0].type, "subtype": premises[0].subtype},
        "premise2": {"form": premises[1].form, "type": premises[1].type, "subtype": premises[1].subtype},
        "premise3": {"form": premises[2].form, "type": premises[2].type, "subtype": premises[2].subtype},
        "conclusion": {"form": conclusion.form, "type": conclusion.type, "subtype": conclusion.subtype},
    }
    return argument_data


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
    """Normalize a logical form using the existing helper function."""
    return normalize_logical_form(form)


def find_matching_sentences(form: str, sentence_type: str = None, sentence_subtype: str = None) -> List[Sentence]:
    """Find all sentences in the database that match the given logical form, optionally filtered by type."""
    try:
        query = db.session.query(Sentence)

        if sentence_type:
            query = query.filter(Sentence.type == sentence_type)
        if sentence_subtype:
            query = query.filter(Sentence.subtype == sentence_subtype)

        # Look for exact matches
        matching_forms = query.filter(Sentence.form == form).all()
        if not matching_forms:
            # Normalize the form and look for matches again
            normalized_form = normalize_form(form)
            matching_forms = query.filter(Sentence.form == normalized_form).all()
        return matching_forms

    except Exception as e:
        print(f"Error finding matching sentences: {e}")
        return []


def find_matching_argument(instance, argument):
    """Find a complete matching argument in the database for the given instance."""
    matching_sentences = []

    print("\nLooking for matching argument:")
    for form, original_sentence in zip(instance, argument.values()):
        print(f"\nLooking for matches for form: {form}")
        print(f"Original type: {original_sentence['type']}")
        matches = find_matching_sentences(
            form, sentence_type=original_sentence["type"], sentence_subtype=original_sentence["subtype"]
        )

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

    # 2. Get name placeholders
    name_placeholders = set()
    for pattern in abstract_patterns:
        name_placeholders.update(re.findall(r"\{\$\d+\}", pattern))
    name_placeholders = sorted(list(name_placeholders))
    valid_names = list(lexicon.names.keys())

    # 3. Generate all possible combinations
    all_placeholders = sorted(predicate_alternatives.keys()) + sorted(name_placeholders)
    all_alternatives = [
        predicate_alternatives[p] if p in predicate_alternatives else valid_names for p in all_placeholders
    ]
    valid_mappings = generate_mappings(all_placeholders, all_alternatives)

    # Generate instances
    print("\nGenerating instances...")
    for i, mapping in enumerate(valid_mappings):
        concrete_forms = []
        for pattern in abstract_patterns:
            concrete = pattern
            for placeholder, value in mapping.items():
                concrete = concrete.replace(placeholder, value)
            concrete_forms.append(concrete)

        # Print first 3 instances in detail
        if i < 3:
            print(f"\nInstance {i+1}:")
            for j, form in enumerate(concrete_forms):
                print(f"  Sentence {j+1}: {form}")

        instances.append(concrete_forms)

    print(f"\nTotal instances generated: {len(instances)}")
    return instances


if __name__ == "__main__":
    try:
        # Create output directory if it doesn't exist
        output_dir = "match_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get argument ID from command line or use default
        argument_id = sys.argv[1] if len(sys.argv) > 1 else "75537f4727941e8f"

        # Create output files with argument ID
        output_file = os.path.join(output_dir, f"matches_{argument_id}.json")
        temp_output_file = os.path.join(output_dir, f"matches_{argument_id}_temp.json")

        # Use a test case with forms we know exist
        argument_data = grab_argument_data(db.session, argument_id)
        print("\nOriginal argument:")
        for key, value in argument_data.items():
            print(f"{key}: {value['form']} (type: {value['type']}, subtype: {value['subtype']})")

        # Define the argument pattern to search for
        argument_pattern = [normalize_form(form["form"]) for form in argument_data.values()]

        print("\nNormalized argument pattern:")
        for i, form in enumerate(argument_pattern):
            print(f"Sentence {i+1}: {form}")

        predicates, names = collect_all_predicates_and_names(argument_pattern)
        global_mapping = create_global_mapping(predicates, names)

        print("\nPredicates found:", predicates)
        print("Names found:", names)
        print("Global mapping:", global_mapping)

        # Abstract the patterns
        abstract_patterns = []
        print("\nAbstracting patterns:")
        for i, form in enumerate(argument_pattern):
            print(f"\nOriginal form {i+1}: {form}")
            abstracted = abstract_form(form, global_mapping)
            abstract_patterns.append(abstracted)
            print(f"Abstracted form {i+1}: {abstracted}")

        # Initialize results structure
        results = {
            "argument_id": argument_id,
            "argument_pattern": argument_pattern,
            "matches": [],
            "total_checked": 0,
            "last_checked": None,
        }

        # Generate and check instances
        print("\nGenerating instances...")
        instances = generate_argument_instances(abstract_patterns, global_mapping)
        print(f"\nGenerated {len(instances)} total instances")

        # Try to find matches
        print("\nSearching for matches...")
        matches_found = 0
        for i, instance in enumerate(instances, 1):
            if i % 10 == 0:  # Print progress more frequently
                print(f"Checked {i} instances...")
                # Update and save progress
                results["total_checked"] = i
                results["last_checked"] = instance
                with open(temp_output_file, "w") as f:
                    json.dump(results, f, indent=2)

            matching_sentences = find_matching_argument(instance, argument_data)
            if matching_sentences:
                matches_found += 1
                match_data = {
                    "premises": [{"id": s.id, "form": s.form} for s in matching_sentences[:-1]],
                    "conclusion": {"id": matching_sentences[-1].id, "form": matching_sentences[-1].form},
                }
                results["matches"].append(match_data)

                print(f"\nFound complete argument match #{matches_found}:")
                for j, sentence in enumerate(matching_sentences, 1):
                    print(f"Sentence {j}: {sentence.form} (ID: {sentence.id})")

                # Save results after each match
                with open(temp_output_file, "w") as f:
                    json.dump(results, f, indent=2)

        print(f"\nTotal matches found: {matches_found}")
        print(f"Total instances checked: {len(instances)}")

        # Save final results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Delete temporary file
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)

        print(f"\nResults saved to {output_file}")
        print("Temporary results file deleted")

    except Exception as e:
        print(f"Error during execution: {e}")
