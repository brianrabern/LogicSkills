from Database.DB import db
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_all_tables():
    """Create all tables in the database."""
    try:
        logger.info("Creating all tables...")
        db.create_tables()
        logger.info("Successfully created all tables")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")


if __name__ == "__main__":
    create_all_tables()
