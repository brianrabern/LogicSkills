from Database.DB import db
from Database.models import Argument, Sentence
import json
from collections import Counter

def get_all_argument_sentence_ids():
	"""Get all unique sentence IDs that are involved in arguments (either as premises or conclusions)."""
	session = db.session

	# Get all arguments
	arguments = session.query(Argument).all()

	# Use a set to store unique IDs
	sentence_ids = set()

	# Process each argument
	for arg in arguments:
		# Add conclusion ID
		sentence_ids.add(arg.conclusion_id)
		# Add premise IDs
		sentence_ids.update(int(pid) for pid in arg.premise_ids.split(','))

	# Convert to sorted list
	sentence_ids = sorted(list(sentence_ids))

	print(f"Found {len(sentence_ids)} unique sentence IDs involved in arguments")
	return sentence_ids

def check_counterpart_ids(sentence_ids):
	"""Check which sentence IDs have counterpart IDs."""
	session = db.session

	# Query all sentences with their counterpart IDs and type
	sentences = session.query(Sentence.id, Sentence.counterpart_id, Sentence.type).filter(
		Sentence.id.in_(sentence_ids)
	).all()

	# Create a dictionary mapping sentence IDs to their counterpart IDs and type
	counterpart_map = {s.id: (s.counterpart_id, s.type) for s in sentences}

	# Get types of counterpart sentences
	counterpart_ids = [cid for cid, _ in counterpart_map.values() if cid is not None]
	counterpart_sentences = session.query(Sentence.id, Sentence.type).filter(
		Sentence.id.in_(counterpart_ids)
	).all()
	counterpart_types = {s.id: s.type for s in counterpart_sentences}

	# Count how many have counterparts
	with_counterparts = sum(1 for cid, _ in counterpart_map.values() if cid is not None)
	without_counterparts = sum(1 for cid, _ in counterpart_map.values() if cid is None)

	print(f"\nFound {with_counterparts} sentences with counterpart IDs")
	print(f"Found {without_counterparts} sentences without counterpart IDs")
	print(f"Out of {len(sentence_ids)} total sentences")

	# Count types of sentences without counterparts
	no_counterpart_types = Counter()
	for cid, type_ in counterpart_map.values():
		if cid is None:
			no_counterpart_types[type_] += 1

	print("\nTypes of sentences without counterparts:")
	for type_, count in no_counterpart_types.most_common():
		print(f"{type_}: {count}")

	# Count types of sentences with counterparts and their counterpart types
	with_counterpart_types = Counter()
	counterpart_type_pairs = Counter()
	for sid, (cid, type_) in counterpart_map.items():
		if cid is not None:
			with_counterpart_types[type_] += 1
			counterpart_type = counterpart_types[cid]
			counterpart_type_pairs[(type_, counterpart_type)] += 1

	print("\nTypes of sentences with counterparts:")
	for type_, count in with_counterpart_types.most_common():
		print(f"{type_}: {count}")

	print("\nType pairs (original -> counterpart):")
	for (type1, type2), count in counterpart_type_pairs.most_common():
		print(f"{type1} -> {type2}: {count}")

	# Print some examples
	print("\nExample sentences with counterparts:")
	examples_with = [(sid, cid, type_) for sid, (cid, type_) in counterpart_map.items() if cid is not None][:5]
	for sid, cid, type_ in examples_with:
		print(f"Sentence {sid} ({type_}) has counterpart {cid} ({counterpart_types[cid]})")

	print("\nExample sentences without counterparts:")
	examples_without = [(sid, type_) for sid, (cid, type_) in counterpart_map.items() if cid is None][:5]
	for sid, type_ in examples_without:
		print(f"Sentence {sid} has no counterpart. type={type_}")

	return counterpart_map

def check_arguments_with_all_counterparts():
	"""Check how many arguments have all their sentences with counterparts."""
	session = db.session

	# Get all arguments
	arguments = session.query(Argument).all()

	# Get all sentence IDs and their counterpart status
	all_sentences = session.query(Sentence.id, Sentence.counterpart_id).all()
	counterpart_map = {s.id: s.counterpart_id for s in all_sentences}

	# Count arguments where all sentences have counterparts
	complete_args = 0
	for arg in arguments:
		# Check conclusion
		if counterpart_map.get(arg.conclusion_id) is None:
			continue

		# Check all premises
		premise_ids = [int(pid) for pid in arg.premise_ids.split(',')]
		if all(counterpart_map.get(pid) is not None for pid in premise_ids):
			complete_args += 1

	print(f"\nOut of {len(arguments)} total arguments:")
	print(f"{complete_args} arguments have all sentences with counterparts")
	print(f"{len(arguments) - complete_args} arguments have at least one sentence without a counterpart")

if __name__ == "__main__":
	# Get all sentence IDs involved in arguments
	ids = get_all_argument_sentence_ids()
	print("\nFirst 10 IDs:", ids[:10])
	print("\nLast 10 IDs:", ids[-10:])

	# Save IDs to JSON
	with open("argument_sentence_ids.json", "w") as f:
		json.dump(ids, f)

	# Check for counterpart IDs
	counterpart_map = check_counterpart_ids(ids)

	# Check arguments with all counterparts
	check_arguments_with_all_counterparts()

	# Save counterpart mapping to JSON
	with open("sentence_counterpart_map.json", "w") as f:
		json.dump(counterpart_map, f)
