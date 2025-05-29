from Database.DB import db
from Database.models import Sentence, Argument
from Utils.normalize import unescape_logical_form


def inspect_argument(session, arg_id):
    # Get the argument
    argument = session.query(Argument).filter_by(id=arg_id).first()
    if not argument:
        print(f"Argument {arg_id} not found")
        return

    language = argument.language

    # Get the domain constraint
    domain_constraint = session.query(Sentence).filter_by(type="domain_constraint", language=language).first()

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
    arg_ids = [
        "12c9fe11c61af2ec",
        "1f2e5316ba1c5a03",
        "1f48b9075932cc3a",
        "22c887456b50a7b4",
        "22df87226c908d18",
        "28d20c4969a707b0",
        "3692db5174569ace",
        "3df85171ad457ecc",
        "3f7cca3f432a1df3",
        "4093cb2e90692d72",
        "50b6d887d5ad2c2a",
        "51a464d5fc5cccf5",
        "81cc083d1f1aa9a5",
        "898f50c1aadab461",
        "90117eeaa45f3d99",
        "9b0cb4b7bf09c16b",
        "a3895189b363f302",
        "a70596728882c1b4",
        "a915b996d2ea107b",
        "aef4913dec3bb76f",
        "ce0a838c295c0c57",
        "d1199f40596e8dbe",
        "d70e5503bf55bdd2",
        "d91d7f0574c79831",
        "e89da97d6bb95d63",
        "f32318aa069180dc",
        "f4d8fac9b61898c0",
        "f91e0ddfde6f73d0",
    ]
    # get all arguments from the db
    # args = session.query(Argument.id).all()
    # arg_ids = [arg.id for arg in args]
    for arg_id in arg_ids:
        inspect_argument(session, arg_id)

    session.close()


if __name__ == "__main__":
    main()
