from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, create_engine, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Podcast(Base):
    __tablename__ = 'podcasts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255))
    feed_url = Column(String(512), unique=True, nullable=False)
    muted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    episodes = relationship("Episode", back_populates="podcast")

class Episode(Base):
    __tablename__ = 'episodes'
    
    id = Column(Integer, primary_key=True)
    podcast_id = Column(Integer, ForeignKey('podcasts.id'))
    guid = Column(String(512), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    published_at = Column(DateTime)
    audio_url = Column(String(512))
    transcript = Column(Text)
    vector_embedding = Column(Text)  # JSON string of vector embedding
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)  # When the episode was processed and written to Obsidian
    
    podcast = relationship("Podcast", back_populates="episodes")

def init_db(db_path: str):
    """Initialize the database and create all tables."""
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    return engine
