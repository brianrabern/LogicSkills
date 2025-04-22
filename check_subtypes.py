from Database.DB import DatabaseManager
from sqlalchemy import distinct

# Initialize database connection
db = DatabaseManager("mariadb+mariadbconnector://root:new_password@localhost:3306/arg")

# Query for distinct subtypes
subtypes = db.session.query(
    distinct(db.session.query(Base.metadata.tables["sentences"].c.subtype).subquery().c.subtype)
).all()

print("\nAll sentence subtypes in database:")
for subtype in subtypes:
    print(subtype[0])

db.close()
