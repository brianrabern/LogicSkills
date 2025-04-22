from Database.DB import db
from Database.models import Sentence
from collections import defaultdict


def main():
    try:
        # Get all sentences
        sentences = db.session.query(Sentence).all()
        print(f"Found {len(sentences)} sentences")

        # Count occurrences of each form
        form_counts = defaultdict(list)
        for sentence in sentences:
            form_counts[sentence.form].append(sentence)

        # Find duplicates
        duplicates = {form: sentences for form, sentences in form_counts.items() if len(sentences) > 1}

        # Collect all unique type-subtype pairs
        pair_types = defaultdict(list)

        for form, sentences in duplicates.items():
            # Create a sorted tuple of (type,subtype) pairs for each duplicate group
            pairs = tuple(sorted((s.type, s.subtype) for s in sentences))
            pair_types[pairs].append((form, sentences))

        if duplicates:
            print("\nFound duplicate forms with the following type pairs:")
            for pairs, forms in pair_types.items():
                # Check if this group contains unless/or pairs
                has_unless = any(s.subtype == "unless" for s in forms[0][1])
                has_or = any(s.subtype == "simple" and s.type == "disjunction" for s in forms[0][1])

                if has_unless and has_or:
                    print(f"\nUNLESS/OR PAIR:")
                    print(f"Number of forms with these pairs: {len(forms)}")
                    # Show first example
                    form, sentences = forms[0]
                    print(f"Example form: {form}")
                    for s in sentences:
                        print(f"  ID: {s.id}, Type: {s.type}, Subtype: {s.subtype}")
                        print(f"  Sentence: {s.sentence}")
                else:
                    print(f"\nType pairs: {pairs}")
                    print(f"Number of forms with these pairs: {len(forms)}")
                    # Show first example
                    form, sentences = forms[0]
                    print(f"Example form: {form}")
                    for s in sentences:
                        print(f"  ID: {s.id}, Type: {s.type}, Subtype: {s.subtype}")
                        print(f"  Sentence: {s.sentence}")

            print(f"\nTotal duplicates: {len(duplicates)}")
        else:
            print("\nNo duplicate forms found!")

    except Exception as e:
        print(f"Error finding duplicates: {e}")


if __name__ == "__main__":
    main()
