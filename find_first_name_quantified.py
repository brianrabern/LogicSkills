from Database.DB import DatabaseManager
from Database.models import Base
from sqlalchemy import func
import time

# Initialize database connection
db = DatabaseManager("mariadb+mariadbconnector://root:new_password@localhost:3306/arg")

# Count total sentences after timestamp
count = (
    db.session.query(func.count())
    .select_from(Base.metadata.tables["sentences"])
    .filter(Base.metadata.tables["sentences"].c.time_created > 1745280061)
    .scalar()
)

print(f"\nNumber of sentences added after {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1745280061))}: {count}")

# Get distribution by type/subtype
type_counts = (
    db.session.query(
        Base.metadata.tables["sentences"].c.type, Base.metadata.tables["sentences"].c.subtype, func.count()
    )
    .filter(Base.metadata.tables["sentences"].c.time_created > 1745280061)
    .group_by(Base.metadata.tables["sentences"].c.type, Base.metadata.tables["sentences"].c.subtype)
    .all()
)

print("\nDistribution by type/subtype:")
for type_, subtype, count in type_counts:
    print(f"{type_} - {subtype}: {count}")

db.close()
