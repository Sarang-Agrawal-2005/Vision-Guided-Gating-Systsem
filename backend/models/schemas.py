from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Point(BaseModel):
    x: int
    y: int

class ZoneCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    coordinates: List[Point] = Field(...)
    threshold: int = Field(default=25, ge=10, le=50)
    min_area: int = Field(default=500, ge=100, le=2000)
    motion_frames: int = Field(default=3, ge=1, le=10)
    priority: int = Field(default=1, ge=1, le=5)

class Zone(ZoneCreate):
    id: str
    created_at: datetime

class ZoneUpdate(BaseModel):
    threshold: Optional[int] = Field(None, ge=10, le=50)
    min_area: Optional[int] = Field(None, ge=100, le=2000)
    motion_frames: Optional[int] = Field(None, ge=1, le=10)
    priority: Optional[int] = Field(None, ge=1, le=5)

class VideoUploadResponse(BaseModel):
    filename: str
    size: int
    video_id: str
    message: str

class VideoProcessRequest(BaseModel):
    video_id: str
    zones: List[str]  # Zone IDs to include

class MotionDetectionStatus(BaseModel):
    active: bool
    beam_active: bool
    active_zones: List[str]
    timestamp: datetime

class BeamEvent(BaseModel):
    timestamp: datetime
    action: str  # "STOPPED" or "RESUMED"
    zone: str
    message: str

class SystemStatus(BaseModel):
    total_zones: int
    active_detection: bool
    beam_status: str
    last_event: Optional[BeamEvent]
