from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    JSON,
    Boolean,
    Float,
)
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.ext.declarative import declarative_base
from enum import IntEnum
from time import time

Base = declarative_base()


class Status(IntEnum):
    UNSAT = -1  # logical falsehood
    SAT = 0  # not logical falsehood and not logical truth
    THEOREM = 1  # logical truth


class Sentence(Base):
    __tablename__ = "sentences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sentence = Column(Text)
    type = Column(String(56), default=None)
    subtype = Column(String(56), default=None)
    soa = Column(JSON, default=None)
    form = Column(Text, default=None)
    ast = Column(JSON, default=None)
    base = Column(TINYINT, default=None)
    status = Column(TINYINT, default=None)
    language = Column(String(50), default=None)
    counterpart_id = Column(Integer, default=None)
    time_created = Column(Integer)

    def to_dict(self):
        return {
            "id": self.id,
            "sentence": self.sentence,
            "type": self.type,
            "subtype": self.subtype,
            "soa": self.soa,
            "form": self.form,
            "ast": self.ast,
            "base": self.base,
            "status": self.status,
            "time_created": self.time_created,
            "language": self.language,
            "counterpart_id": self.counterpart_id,
        }


class Argument(Base):
    __tablename__ = "arguments"
    id = Column(String(64), primary_key=True)
    premise_ids = Column(String(255), nullable=False)
    conclusion_id = Column(Integer, nullable=False)
    valid = Column(Boolean, nullable=False)
    difficulty = Column(Float, default=0.0)
    source = Column(String(50))
    created_at = Column(Integer, default=lambda: int(time()))
    language = Column(String(50), default=None)
