from Database.DB import db
from Database.models import Sentence
from Utils.normalize import unescape_logical_form


def inspect_sentences(session, premise_ids, conclusion_id):
    # Get the domain constraint
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint").first()

    # Get the premises
    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()
    if len(premises) != len(premise_ids):
        print(f"Error: Expected {len(premise_ids)} premises, got {len(premises)}")
        return

    # Get the conclusion
    conclusion = session.query(Sentence).filter_by(id=conclusion_id).first()
    if not conclusion:
        print(f"Error: Conclusion {conclusion_id} not found")
        return

    print("\n=== Sentences Inspection ===")
    print(f"Premise IDs: {premise_ids}")
    print(f"Conclusion ID: {conclusion_id}")

    print("\nPremises:")
    for premise in premises:
        print(f"\n{premise.id}: {premise.sentence}")
    print("\nConclusion:")
    print(f"{conclusion.id}: {conclusion.sentence}")
    print("\nLogical Form:\n")

    # Build the logical form string
    premise_forms = [unescape_logical_form(p.form) for p in premises]
    domain_form = unescape_logical_form(domain_constraint.form) if domain_constraint else ""
    conclusion_form = unescape_logical_form(conclusion.form)

    logical_form = f"{domain_form + ', ' if domain_form else ''}{', '.join(premise_forms)} |= {conclusion_form}"
    print(logical_form)

    print("=" * 50)


def main():
    session = db.Session()

    # Example usage - replace these with your desired sentence IDs
    premise_ids = [
        "763",
        "604",
        "24018",
        "2755",  # Added more premises
        "19620",
    ]
    conclusion_id = "753"

    inspect_sentences(session, premise_ids, conclusion_id)
    session.close()


if __name__ == "__main__":
    main()
