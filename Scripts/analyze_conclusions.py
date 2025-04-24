from Database.DB import db
from Database.models import Argument, Sentence
from sqlalchemy import desc
from collections import defaultdict


def analyze_ast_type(ast):
    """Analyze the type of an AST node"""
    if not isinstance(ast, list):
        return "atomic"

    node_type = ast[0]
    if node_type == "forall":
        return "universal"
    elif node_type == "exists":
        return "existential"
    elif node_type == "implies":
        return "conditional"
    elif node_type == "or":
        return "disjunction"
    elif node_type == "and":
        return "conjunction"
    elif node_type == "not":
        return "negation"
    else:
        return "atomic"


def count_nested_conditionals(ast):
    """Count nested conditionals in an AST"""
    if not isinstance(ast, list):
        return 0
    count = 0
    if ast[0] in ["implies", "or"]:
        for child in ast[1:]:
            if isinstance(child, list) and child[0] in ["implies", "or"]:
                count += 1
    for child in ast[1:]:
        count += count_nested_conditionals(child)
    return count


def main():
    # Get all valid arguments ordered by creation time
    valid_args = db.session.query(Argument).filter_by(valid=True).order_by(desc(Argument.created_at)).all()

    # Initialize counters
    type_counts = defaultdict(int)
    subtype_counts = defaultdict(int)
    type_subtype_counts = defaultdict(lambda: defaultdict(int))

    # Analyze each conclusion
    for arg in valid_args:
        conclusion = db.session.query(Sentence).get(arg.conclusion_id)
        if not conclusion:
            continue

        # Count types and subtypes
        type_counts[conclusion.type] += 1
        if conclusion.subtype:
            subtype_counts[conclusion.subtype] += 1
            type_subtype_counts[conclusion.type][conclusion.subtype] += 1

    # Print results
    print("\nConclusion Type Distribution:")
    print("----------------------------")
    total = len(valid_args)
    for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{type_name}: {count} ({percentage:.1f}%)")

        # Print subtypes for this type
        if type_name in type_subtype_counts:
            print("  Subtypes:")
            for subtype, subcount in sorted(type_subtype_counts[type_name].items(), key=lambda x: x[1], reverse=True):
                subpercentage = (subcount / count) * 100
                print(f"    {subtype}: {subcount} ({subpercentage:.1f}%)")

    # Print some example conclusions
    print("\nExample Conclusions:")
    print("-------------------")
    for arg in valid_args[:5]:  # Show first 5
        conclusion = db.session.query(Sentence).get(arg.conclusion_id)
        if conclusion:
            print(f"ID: {conclusion.id}")
            print(f"Type: {conclusion.type}")
            print(f"Subtype: {conclusion.subtype}")
            print(f"Text: {conclusion.form}")
            print("---")


if __name__ == "__main__":
    main()
