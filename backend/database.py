from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Database URL - using SQLite
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///./data/motion_detection.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import Base here to avoid circular imports
from models import Base

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
