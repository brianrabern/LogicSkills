from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from Database.models import Base, Sentence, Argument
from Utils.normalize import normalize_logical_form, unescape_logical_form
from config import mariadb_url
from sqlalchemy.exc import IntegrityError


class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        print("Connected to the database", db_url)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(self.engine)

    def recreate_tables(self):
        self.drop_tables()
        self.create_tables()

    def add_sentence(
        self,
        sentence,
        type,
        subtype,
        soa,
        form,
        ast,
        base,
        status,
        language,
        counterpart_id,
        time_created,
    ):
        normalized_form = normalize_logical_form(form)
        s = Sentence(
            sentence=sentence,
            type=type,
            subtype=subtype,
            soa=soa,
            form=normalized_form,
            ast=ast,
            base=base,
            status=status,
            language=language,
            counterpart_id=counterpart_id,
            time_created=time_created,
        )
        self.session.add(s)
        self.session.commit()

    def sentence_exists(self, form, type=None, subtype=None, language=None):
        normalized_form = normalize_logical_form(form)
        query = self.session.query(Sentence).filter_by(form=normalized_form)

        if type is not None:
            query = query.filter_by(type=type)
        if subtype is not None:
            query = query.filter_by(subtype=subtype)
        if language is not None:
            query = query.filter_by(language=language)

        return query.first() is not None

    def update_record(self, model, record_id, **kwargs):
        try:
            result = self.session.query(model).filter_by(id=record_id).update(kwargs)
            self.session.commit()
            return result > 0
        except Exception as e:
            print(f"Error updating record: {e}")
            self.session.rollback()
            return False

    def get_all_sentences(self):
        sentence_objs = self.session.query(Sentence).all()
        sentences = [s.to_dict() for s in sentence_objs]
        for s in sentences:
            s["form"] = unescape_logical_form(s["form"])
        return sentences

    def get_base_entries(self, language=None):
        query = self.session.query(Sentence).filter_by(base=True)
        if language is not None:
            query = query.filter_by(language=language)
        sentence_objs = query.all()
        sentences = [s.to_dict() for s in sentence_objs]
        for s in sentences:
            s["form"] = unescape_logical_form(s["form"])
        return sentences

    def get_sentence_where(self, **kwargs):
        """
        Get sentences based on the provided keyword arguments.
        Example: get_sentence_where(type="conditional", base=True, language="english")
        """
        sentence_objs = self.session.query(Sentence).filter_by(**kwargs).all()
        sentences = [s.to_dict() for s in sentence_objs]
        for s in sentences:
            s["form"] = unescape_logical_form(s["form"])
        return sentences

    def get_random_sentence(self, exclude_status=None, types=None):
        query = self.session.query(Sentence)
        if exclude_status is not None:
            query = query.filter(Sentence.status != exclude_status)
        if types:
            query = query.filter(Sentence.type.in_(types))
        res = query.order_by(func.rand()).limit(1).all()
        sen = res[0]
        sen["form"] = unescape_logical_form(sen["form"])
        return sen

    def get_last_inserted_id(self):
        """Get the ID of the last inserted sentence."""
        try:
            return self.session.query(Sentence).order_by(Sentence.id.desc()).first().id
        except Exception as e:
            print(f"Error getting last inserted ID: {e}")
            return None

    def update_sentence_counterpart(self, sentence_id, counterpart_id):
        """Update a sentence's counterpart_id."""
        try:
            result = self.session.query(Sentence).filter_by(id=sentence_id).update({"counterpart_id": counterpart_id})
            self.session.commit()
            return result > 0
        except Exception as e:
            print(f"Error updating sentence counterpart: {e}")
            self.session.rollback()
            return False

    def add_argument(self, arg_id, premise_ids, conclusion_id, valid, difficulty=None, source="z3", language=None):
        """Add a new argument to the database."""
        try:
            argument = Argument(
                id=arg_id,
                premise_ids=premise_ids,
                conclusion_id=conclusion_id,
                valid=valid,
                difficulty=difficulty,
                source=source,
                language=language,
            )
            self.session.add(argument)
            self.session.commit()
            return True
        except IntegrityError:
            self.session.rollback()
            return False
        except Exception as e:
            print(f"Error adding argument: {e}")
            self.session.rollback()
            return False

    def get_argument(self, arg_id):
        """Get an argument by its ID."""
        try:
            return self.session.query(Argument).filter_by(id=arg_id).first()
        except Exception as e:
            print(f"Error getting argument: {e}")
            return None

    def get_argument_where(self, **kwargs):
        """
        Get arguments based on the provided keyword arguments.
        Example: get_argument_where(premise_ids="1,2,3", valid=True, language="english")
        """
        try:
            return self.session.query(Argument).filter_by(**kwargs).first()
        except Exception as e:
            print(f"Error getting argument: {e}")
            return None

    def get_valid_argument_count(self, language):
        """Get the count of valid arguments for a given language."""
        try:
            return self.session.query(Argument).filter_by(valid=True, language=language).count()
        except Exception as e:
            print(f"Error getting valid argument count: {e}")
            return 0

    def close(self):
        self.session.close()


db = DatabaseManager(mariadb_url)
