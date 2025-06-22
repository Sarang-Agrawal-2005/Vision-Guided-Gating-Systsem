from sqlalchemy import String, Integer, Float, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from datetime import datetime
import json
from typing import Dict, Any, Optional

class Base(DeclarativeBase):
    pass

class Zone(Base):
    __tablename__ = "zones"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    coordinates: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[int] = mapped_column(Integer, nullable=False)
    min_area: Mapped[int] = mapped_column(Integer, nullable=False)
    motion_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    color: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "coordinates": json.loads(self.coordinates),
            "priority": self.priority,
            "threshold": self.threshold,
            "min_area": self.min_area,
            "motion_frames": self.motion_frames,
            "color": self.color,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Video(Base):
    __tablename__ = "videos"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    original_name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "original_name": self.original_name,
            "path": self.path,
            "size": self.size,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "content_type": self.content_type,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None
        }
