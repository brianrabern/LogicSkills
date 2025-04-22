from Database.DB import db
from Database.models import Argument, Sentence


def inspect_arguments():
    # Query all valid arguments
    valid_args = db.session.query(Argument).filter_by(valid=True).all()

    print(f"\nFound {len(valid_args)} valid arguments\n")

    for arg in valid_args:
        print(f"Argument ID: {arg.id}")
        print(f"Difficulty: {arg.difficulty}")
        print(f"Source: {arg.source}")
        print("\nPremises:")

        # Get premise sentences
        premise_ids = [int(pid) for pid in arg.premise_ids.split(",")]
        premises = db.session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()

        for i, premise in enumerate(premises, 1):
            print(f"{i}. {premise.sentence}")
            print(f"   Logical Form: {premise.form}")

        # Get conclusion sentence
        conclusion = db.session.query(Sentence).get(arg.conclusion_id)
        print("\nConclusion:")
        print(f"{conclusion.sentence}")
        print(f"Logical Form: {conclusion.form}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    inspect_arguments()
