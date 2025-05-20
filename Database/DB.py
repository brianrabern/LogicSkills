from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from Database.models import Base, Name, Predicate, Sentence
from Utils.normalize import normalize_logical_form, unescape_logical_form
from config import mariadb_url


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

    def add_name(self, symbol, name, gender, time_created):
        name_obj = Name(symbol=symbol, name=name, gender=gender, time_created=time_created)
        self.session.add(name_obj)
        self.session.commit()

    def add_predicate(
        self,
        symbol,
        template,
        negated_template,
        arity,
        structure,
        semantic_type,
        tense,
        time_created,
    ):
        pred = Predicate(
            symbol=symbol,
            template=template,
            negated_template=negated_template,
            arity=arity,
            structure=structure,
            semantic_type=semantic_type,
            tense=tense,
            time_created=time_created,
        )
        self.session.add(pred)
        self.session.commit()

    def add_sentence(
        self, sentence, type, subtype, soa, form, ast, base, status, time_created, language, counterpart_id=None
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
            time_created=time_created,
            language=language,
            counterpart_id=counterpart_id,
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

    def close(self):
        self.session.close()


db = DatabaseManager(mariadb_url)
