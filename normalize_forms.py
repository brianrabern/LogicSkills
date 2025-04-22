from Database.DB import db
from Database.models import Sentence
from Utils.normalize import normalize_logical_form


def main():
    try:
        count = 0
        # Query directly to see raw forms
        sentences = db.session.query(Sentence).all()
        print(f"Found {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            old_form = sentence.form  # Access form directly from SQLAlchemy object
            new_form = normalize_logical_form(old_form)
            print(f"Forms equal: {old_form == new_form}")

            if old_form != new_form:
                print(f"Would update form: {old_form} -> {new_form}")
                db.update_record(Sentence, sentence.id, form=new_form)
                count += 1

        print("\nDatabase forms have been normalized to escaped Unicode format.")
        print(f"Total forms updated: {count}")
    except Exception as e:
        print(f"Error updating database forms: {e}")


if __name__ == "__main__":
    main()
