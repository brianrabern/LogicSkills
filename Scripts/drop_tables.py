from Database.DB import db
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_all_tables():
    """Drop all tables in the database."""
    try:
        logger.info("Dropping all tables...")
        db.drop_tables()
        logger.info("Successfully dropped all tables")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")


if __name__ == "__main__":
    drop_all_tables()
