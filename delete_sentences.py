from Database.DB import DatabaseManager
from Database.models import Base
from sqlalchemy import func

# Initialize database connection
db = DatabaseManager("mariadb+mariadbconnector://root:new_password@localhost:3306/arg")


def get_counts():
    # Get counts for each type/subtype after timestamp
    type_counts = (
        db.session.query(
            Base.metadata.tables["sentences"].c.type, Base.metadata.tables["sentences"].c.subtype, func.count()
        )
        .filter(Base.metadata.tables["sentences"].c.time_created > 1745280061)
        .group_by(Base.metadata.tables["sentences"].c.type, Base.metadata.tables["sentences"].c.subtype)
        .all()
    )
    return type_counts


# Get counts before deletion
print("\nCounts before deletion:")
for type_, subtype, count in get_counts():
    print(f"{type_} - {subtype}: {count}")

# Delete sentences matching criteria
delete_query = (
    db.session.query(Base.metadata.tables["sentences"])
    .filter(Base.metadata.tables["sentences"].c.time_created > 1745280061)
    .filter(
        (
            (Base.metadata.tables["sentences"].c.type == "conjunction")
            & (Base.metadata.tables["sentences"].c.subtype == "contrastive")
        )
        | (
            (Base.metadata.tables["sentences"].c.type == "disjunction")
            & (Base.metadata.tables["sentences"].c.subtype == "negated_disjunct")
        )
    )
)

# Count before deletion
count_before = delete_query.count()
print(f"\nFound {count_before} sentences to delete")

# Delete the sentences
delete_query.delete(synchronize_session=False)
db.session.commit()

print(f"Deleted {count_before} sentences")

# Get counts after deletion
print("\nCounts after deletion:")
for type_, subtype, count in get_counts():
    print(f"{type_} - {subtype}: {count}")

db.close()
