from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, create_tables
from models import Zone as ZoneModel, Base
import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import aiofiles
from pathlib import Path
from PIL import Image
import io
import logging
import json

# Add these imports to your existing main.py
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Motion Detection Beam Control System",
    description="Radiotherapy motion tracking and beam control API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
        # Remove "*" for production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Create necessary directories
os.makedirs("uploads", exist_ok=True)

# In-memory storage for videos (keeping this for now)
videos_db: Dict[str, dict] = {}

# Database startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    logger.info("Database tables created/verified")

# Zone data models
class ZoneCoordinate(BaseModel):
    x: float
    y: float

class Zone(BaseModel):
    id: Optional[str] = None
    name: str
    coordinates: List[ZoneCoordinate]
    priority: int
    threshold: int
    min_area: int
    motion_frames: int
    color: str
    created_at: Optional[str] = None

class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    coordinates: Optional[List[ZoneCoordinate]] = None
    priority: Optional[int] = None
    threshold: Optional[int] = None
    min_area: Optional[int] = None
    motion_frames: Optional[int] = None
    color: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "Motion Detection Beam Control System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Video endpoints (keeping existing functionality)
@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for baseline establishment with enhanced validation"""
    logger.info(f"Uploading video: {file.filename}")
    
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate content type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Read file content
    content = await file.read()
    
    # Check file size (100MB limit)
    max_size = 100 * 1024 * 1024  # 100MB
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    
    # Validate file extension
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    filename = f"{video_id}{file_extension}"
    file_path = f"uploads/{filename}"
    
    # Save uploaded file
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        logger.info(f"File saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Process video and extract metadata
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {file_path}")
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate video properties
        if frame_count == 0 or width == 0 or height == 0 or fps == 0:
            cap.release()
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid video file or corrupted")
        
        # Test if we can read the first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Cannot read video frames")
        
        cap.release()
        logger.info(f"Video processed successfully: {width}x{height}, {frame_count} frames, {fps} fps")
        
    except cv2.error as e:
        logger.error(f"OpenCV error: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"OpenCV error processing video: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    # Store video metadata
    videos_db[video_id] = {
        "id": video_id,
        "filename": filename,
        "original_name": file.filename,
        "path": file_path,
        "size": len(content),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "uploaded_at": datetime.now().isoformat(),
        "content_type": file.content_type
    }
    
    logger.info(f"Video uploaded successfully with ID: {video_id}")
    return {
        "filename": file.filename,
        "size": len(content),
        "video_id": video_id,
        "message": "Video uploaded and processed successfully"
    }

@app.get("/api/video/{video_id}")
async def get_video_info(video_id: str):
    """Get video metadata"""
    logger.info(f"Getting video info for: {video_id}")
    
    if video_id not in videos_db:
        logger.error(f"Video not found: {video_id}")
        raise HTTPException(status_code=404, detail="Video not found")
    
    return videos_db[video_id]

@app.get("/api/video/{video_id}/first-frame")
async def get_video_first_frame(video_id: str):
    """Get the first frame of a video as an image for baseline establishment"""
    logger.info(f"Getting first frame for video: {video_id}")
    
    if video_id not in videos_db:
        logger.error(f"Video not found in database: {video_id}")
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_info = videos_db[video_id]
    video_path = video_info["path"]
    
    logger.info(f"Video path: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found on disk: {video_path}")
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        # Open video and get first frame
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Could not read first frame from: {video_path}")
            raise HTTPException(status_code=400, detail="Could not read video frame")
        
        logger.info(f"Successfully read frame with shape: {frame.shape}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        
        logger.info(f"Successfully created JPEG image, size: {len(img_byte_arr.getvalue())} bytes")
        
        return StreamingResponse(
            io.BytesIO(img_byte_arr.read()),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}
        )
        
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting frame: {str(e)}")

@app.get("/api/videos")
async def list_videos():
    """Get list of all uploaded videos"""
    return list(videos_db.values())

@app.delete("/api/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and its file"""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_info = videos_db[video_id]
    file_path = video_info["path"]
    
    # Remove file if it exists
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
    # Remove from database
    del videos_db[video_id]
    
    return {"message": f"Video '{video_info['original_name']}' deleted successfully"}

# Updated Zone Endpoints with Database Integration
@app.post("/api/zones")
async def create_zone(zone: Zone, db: Session = Depends(get_db)):
    """Create a new detection zone with database persistence"""
    logger.info(f"Creating zone: {zone.name}")
    
    # Generate ID if not provided
    if not zone.id:
        zone.id = str(uuid.uuid4())
    
    # Check for duplicate names
    existing_zone = db.query(ZoneModel).filter(ZoneModel.name == zone.name).first()
    if existing_zone:
        raise HTTPException(status_code=400, detail="Zone name already exists")
    
    # Validate coordinates
    if len(zone.coordinates) < 3:
        raise HTTPException(status_code=400, detail="Zone must have at least 3 coordinates")
    
    # Validate priority
    if zone.priority < 1 or zone.priority > 5:
        raise HTTPException(status_code=400, detail="Priority must be between 1 and 5")
    
    # Create database record
    db_zone = ZoneModel(
        id=zone.id,
        name=zone.name,
        coordinates=json.dumps([{"x": coord.x, "y": coord.y} for coord in zone.coordinates]),
        priority=zone.priority,
        threshold=zone.threshold,
        min_area=zone.min_area,
        motion_frames=zone.motion_frames,
        color=zone.color
    )
    
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    
    logger.info(f"Zone created successfully with ID: {zone.id}")
    return {
        "id": zone.id,
        "message": f"Zone '{zone.name}' created successfully"
    }

@app.get("/api/zones")
async def list_zones(db: Session = Depends(get_db)):
    """Get list of all zones from database"""
    zones = db.query(ZoneModel).all()
    return [zone.to_dict() for zone in zones]

@app.get("/api/zones/{zone_id}")
async def get_zone(zone_id: str, db: Session = Depends(get_db)):
    """Get specific zone by ID"""
    zone = db.query(ZoneModel).filter(ZoneModel.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone.to_dict()

@app.put("/api/zones/{zone_id}")
async def update_zone(zone_id: str, zone_update: ZoneUpdate, db: Session = Depends(get_db)):
    """Update existing zone"""
    zone = db.query(ZoneModel).filter(ZoneModel.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    # Update fields if provided
    if zone_update.name is not None:
        # Check for duplicate names (excluding current zone)
        existing = db.query(ZoneModel).filter(
            ZoneModel.name == zone_update.name,
            ZoneModel.id != zone_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Zone name already exists")
        zone.name = zone_update.name
    
    if zone_update.coordinates is not None:
        if len(zone_update.coordinates) < 3:
            raise HTTPException(status_code=400, detail="Zone must have at least 3 coordinates")
        zone.coordinates = json.dumps([{"x": coord.x, "y": coord.y} for coord in zone_update.coordinates])
    
    if zone_update.priority is not None:
        if zone_update.priority < 1 or zone_update.priority > 5:
            raise HTTPException(status_code=400, detail="Priority must be between 1 and 5")
        zone.priority = zone_update.priority
    
    if zone_update.threshold is not None:
        zone.threshold = zone_update.threshold
    
    if zone_update.min_area is not None:
        zone.min_area = zone_update.min_area
    
    if zone_update.motion_frames is not None:
        zone.motion_frames = zone_update.motion_frames
    
    if zone_update.color is not None:
        zone.color = zone_update.color
    
    zone.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(zone)
    
    logger.info(f"Zone updated: {zone_id}")
    return {
        "id": zone_id,
        "message": f"Zone '{zone.name}' updated successfully"
    }

@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: str, db: Session = Depends(get_db)):
    """Delete zone"""
    zone = db.query(ZoneModel).filter(ZoneModel.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    zone_name = zone.name
    db.delete(zone)
    db.commit()
    
    logger.info(f"Zone deleted: {zone_id}")
    return {
        "message": f"Zone '{zone_name}' deleted successfully"
    }

@app.get("/api/zones/export")
async def export_zones(db: Session = Depends(get_db)):
    """Export all zones configuration"""
    zones = db.query(ZoneModel).all()
    config = {
        "zones": [zone.to_dict() for zone in zones],
        "exported_at": datetime.now().isoformat(),
        "version": "1.0"
    }
    
    return StreamingResponse(
        io.BytesIO(json.dumps(config, indent=2).encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=zones_config.json"}
    )

@app.post("/api/zones/import")
async def import_zones(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Import zones configuration"""
    if not file.filename or not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON file")
    
    try:
        content = await file.read()
        config = json.loads(content.decode())
        
        if "zones" not in config or not isinstance(config["zones"], list):
            raise HTTPException(status_code=400, detail="Invalid configuration file format")
        
        imported_count = 0
        for zone_data in config["zones"]:
            # Validate zone data
            if not all(key in zone_data for key in ["name", "coordinates", "priority"]):
                continue
            
            # Generate new ID to avoid conflicts
            zone_id = str(uuid.uuid4())
            
            # Create database record
            db_zone = ZoneModel(
                id=zone_id,
                name=f"{zone_data['name']}_imported",  # Avoid name conflicts
                coordinates=json.dumps(zone_data["coordinates"]),
                priority=zone_data.get("priority", 1),
                threshold=zone_data.get("threshold", 50),
                min_area=zone_data.get("min_area", 100),
                motion_frames=zone_data.get("motion_frames", 3),
                color=zone_data.get("color", "#FF0000")
            )
            
            db.add(db_zone)
            imported_count += 1
        
        db.commit()
        
        return {
            "message": f"Successfully imported {imported_count} zones",
            "imported_count": imported_count
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing zones: {str(e)}")

@app.post("/api/zones/validate")
async def validate_zone_coordinates(coordinates: List[ZoneCoordinate]):
    """Validate zone coordinates"""
    if len(coordinates) < 3:
        raise HTTPException(status_code=400, detail="Zone must have at least 3 coordinates")
    
    # Calculate area using shoelace formula
    area = 0
    n = len(coordinates)
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i].x * coordinates[j].y
        area -= coordinates[j].x * coordinates[i].y
    area = abs(area) / 2
    
    return {
        "valid": True,
        "area": area,
        "perimeter": calculate_perimeter(coordinates),
        "points_count": len(coordinates)
    }

def calculate_perimeter(coordinates: List[ZoneCoordinate]) -> float:
    """Calculate perimeter of polygon"""
    perimeter = 0
    n = len(coordinates)
    for i in range(n):
        j = (i + 1) % n
        dx = coordinates[j].x - coordinates[i].x
        dy = coordinates[j].y - coordinates[i].y
        perimeter += (dx * dx + dy * dy) ** 0.5
    return perimeter

def format_duration(seconds):
    """Format duration in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "videos_uploaded": len(videos_db)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
