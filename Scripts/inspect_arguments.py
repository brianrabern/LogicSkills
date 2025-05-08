from Database.DB import db
from Database.models import Sentence, Argument
from Utils.normalize import unescape_logical_form


def inspect_argument(session, arg_id):
    # Get the argument
    argument = session.query(Argument).filter_by(id=arg_id).first()
    if not argument:
        print(f"Argument {arg_id} not found")
        return

    # Get the domain constraint
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint").first()

    # Get the premises
    premise_ids = argument.premise_ids.split(",")
    premises = session.query(Sentence).filter(Sentence.id.in_(premise_ids)).all()

    # Get the conclusion
    conclusion = session.query(Sentence).filter_by(id=argument.conclusion_id).first()

    print(f"\n=== Argument {arg_id} ===")
    print(f"Valid: {argument.valid}")
    print(f"Source: {argument.source}")
    print(f"Difficulty: {argument.difficulty}")

    print("\nPremises:")
    for premise in premises:
        print(f"\n{premise.id}: {premise.sentence}")
    print("\nConclusion:")
    print(f"{conclusion.id}: {conclusion.sentence}")
    print("\nLogical Form:\n")
    print(
        f"{unescape_logical_form(domain_constraint.form)}, {unescape_logical_form(premises[0].form)}, {unescape_logical_form(premises[1].form)}, {unescape_logical_form(premises[2].form)} |= {unescape_logical_form(conclusion.form)}"
    )

    print("=" * 50)


def main():
    session = db.Session()

    # List of argument IDs to inspect
    # arg_ids = ["d64962fa5dd09a5b", "7e8c3f2cffef3712", "f49538837b150f8a", "d75652bedaf921ae"]
    # get all arguments from the db
    args = session.query(Argument.id).all()
    arg_ids = [arg.id for arg in args]
    for arg_id in arg_ids:
        inspect_argument(session, arg_id)

    session.close()


if __name__ == "__main__":
    main()
