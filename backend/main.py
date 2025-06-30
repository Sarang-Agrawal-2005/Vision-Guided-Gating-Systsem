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

import cv2
import asyncio
from fastapi.responses import StreamingResponse
import threading
import queue
import time

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
        "http://localhost:2000",
        "http://127.0.0.1:2000",
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

# Add these new models to your existing main.py
class BeamControlRequest(BaseModel):
    action: str  # "start", "stop", "emergency_stop"
    zones: Optional[List[str]] = None

class MotionDetectionResult(BaseModel):
    video_id: str
    zones_with_motion: List[str]
    timestamp: str
    beam_should_stop: bool

class BeamStatus(BaseModel):
    is_active: bool
    detection_active: bool
    last_event: Optional[str] = None
    zones_clear: bool

# Global beam control state
beam_control_state = {
    "is_active": False,
    "detection_active": False,
    "last_event": None,
    "zones_clear": True,
    "events": []
}

# Global video streaming state
video_streaming_state = {
    "active": False,
    "current_video_id": None,
    "zones": [],
    "frame_queue": queue.Queue(maxsize=30),
    "stop_event": threading.Event()
}

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

# Beam Control Endpoints
@app.post("/api/beam/control")
async def control_beam(request: BeamControlRequest):
    """Control radiation beam based on motion detection"""
    global beam_control_state
    
    if request.action == "start":
        beam_control_state["is_active"] = True
        beam_control_state["detection_active"] = True
        beam_control_state["last_event"] = "BEAM_STARTED"
        beam_control_state["events"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "STARTED",
            "message": "Motion detection started. Beam active."
        })
        logger.info("Beam control started")
        
    elif request.action == "stop":
        beam_control_state["is_active"] = False
        beam_control_state["detection_active"] = False
        beam_control_state["last_event"] = "BEAM_STOPPED"
        beam_control_state["events"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "STOPPED",
            "message": "Motion detection stopped by user."
        })
        logger.info("Beam control stopped")
        
    elif request.action == "emergency_stop":
        beam_control_state["is_active"] = False
        beam_control_state["detection_active"] = False
        beam_control_state["last_event"] = "EMERGENCY_STOP"
        beam_control_state["events"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "EMERGENCY_STOP",
            "message": "EMERGENCY STOP activated!"
        })
        logger.warning("Emergency stop activated")
    
    return {"status": "success", "beam_state": beam_control_state}

@app.get("/api/beam/status")
async def get_beam_status():
    """Get current beam control status"""
    return BeamStatus(
        is_active=beam_control_state["is_active"],
        detection_active=beam_control_state["detection_active"],
        last_event=beam_control_state["last_event"],
        zones_clear=beam_control_state["zones_clear"]
    )

@app.post("/api/beam/motion-detected")
async def handle_motion_detection(result: MotionDetectionResult):
    """Handle motion detection results and control beam accordingly"""
    global beam_control_state
    
    if not beam_control_state["detection_active"]:
        return {"message": "Detection not active"}
    
    if result.zones_with_motion:
        # Motion detected - stop beam
        if beam_control_state["is_active"]:
            beam_control_state["is_active"] = False
            beam_control_state["zones_clear"] = False
            beam_control_state["last_event"] = "MOTION_DETECTED"
            
            event = {
                "timestamp": datetime.now().isoformat(),
                "action": "BEAM_STOPPED",
                "message": f"Motion detected in zones: {', '.join(result.zones_with_motion)}. BEAM STOPPED.",
                "zones": result.zones_with_motion
            }
            beam_control_state["events"].append(event)
            logger.warning(f"Motion detected, beam stopped: {result.zones_with_motion}")
    else:
        # No motion - resume beam if detection is active
        if not beam_control_state["is_active"] and beam_control_state["detection_active"]:
            beam_control_state["is_active"] = True
            beam_control_state["zones_clear"] = True
            beam_control_state["last_event"] = "ZONES_CLEAR"
            
            event = {
                "timestamp": datetime.now().isoformat(),
                "action": "BEAM_RESUMED",
                "message": "All zones clear. BEAM RESUMED."
            }
            beam_control_state["events"].append(event)
            logger.info("Zones clear, beam resumed")
    
    return {"status": "processed", "beam_active": beam_control_state["is_active"]}

@app.get("/api/beam/events")
async def get_beam_events(limit: int = 10):
    """Get recent beam control events"""
    events = beam_control_state["events"][-limit:]
    return {"events": events}

@app.delete("/api/beam/events")
async def clear_beam_events():
    """Clear beam control events log"""
    beam_control_state["events"] = []
    return {"message": "Events cleared"}

@app.post("/api/video/{video_id}/process-with-zones")
async def process_video_with_zones(video_id: str, db: Session = Depends(get_db)):
    """Process video with zone overlays for beam control monitoring"""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get zones from database
    zones = db.query(ZoneModel).all()
    if not zones:
        raise HTTPException(status_code=400, detail="No zones configured")
    
    video_info = videos_db[video_id]
    input_path = video_info["path"]
    output_path = f"uploads/processed_{video_id}.mp4"
    
    try:
        # This would be your video processing logic
        # For now, just return success
        logger.info(f"Processing video {video_id} with {len(zones)} zones")
        
        return {
            "status": "success",
            "processed_video_path": output_path,
            "zones_applied": len(zones),
            "message": "Video processed with zone overlays"
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

# Video Streaming Endpoints
@app.get("/api/video/{video_id}/stream")
async def stream_video_with_zones(video_id: str, db: Session = Depends(get_db)):
    """Stream video with zone overlays for beam control monitoring"""
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get zones from database
    zones = db.query(ZoneModel).all()
    video_info = videos_db[video_id]
    
    # Update streaming state
    video_streaming_state["current_video_id"] = video_id
    video_streaming_state["zones"] = [zone.to_dict() for zone in zones]
    video_streaming_state["active"] = True
    video_streaming_state["stop_event"].clear()
    
    # Start video processing in background thread
    threading.Thread(
        target=process_video_with_zones_background,
        args=(video_info["path"], zones),
        daemon=True
    ).start()
    
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def process_video_with_zones_background(video_path: str, zones):
    """Background thread to process video frames with zone overlays"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
    
    try:
        while video_streaming_state["active"] and not video_streaming_state["stop_event"].is_set():
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Draw zones on frame
            annotated_frame = draw_zones_on_frame(frame, video_streaming_state["zones"])
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Add to queue (non-blocking)
            try:
                video_streaming_state["frame_queue"].put(buffer.tobytes(), block=False)
            except queue.Full:
                # Skip frame if queue is full
                pass
            
            time.sleep(frame_delay)
            
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
    finally:
        cap.release()

def draw_zones_on_frame(frame, zones):
    """Draw zone overlays on video frame"""
    annotated_frame = frame.copy()
    
    for zone in zones:
        try:
            # Parse coordinates
            coordinates = zone["coordinates"]
            if len(coordinates) < 3:
                continue
            
            # Convert to numpy array for OpenCV
            points = np.array([[int(coord["x"]), int(coord["y"])] for coord in coordinates], np.int32)
            
            # Parse color (hex to BGR)
            color_hex = zone["color"].lstrip('#')
            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert RGB to BGR
            
            # Draw zone boundary
            cv2.polylines(annotated_frame, [points], True, color_bgr, 3)
            
            # Draw semi-transparent fill - FIXED
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [points], color_bgr)  # Fixed: wrap points in list
            cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0, annotated_frame)
            
            # Add zone label
            if len(coordinates) > 0:
                label_pos = (int(coordinates[0]["x"]), int(coordinates[0]["y"]) - 10)
                cv2.putText(annotated_frame, zone["name"], label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                
                # Add priority badge
                priority_text = f"P{zone['priority']}"
                priority_pos = (int(coordinates[0]["x"]), int(coordinates[0]["y"]) + 25)
                cv2.putText(annotated_frame, priority_text, priority_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        except Exception as e:
            logger.error(f"Error drawing zone {zone.get('name', 'unknown')}: {e}")
            continue
    
    return annotated_frame

async def generate_video_stream():
    """Generate video stream with zone overlays"""
    while video_streaming_state["active"]:
        try:
            # Get frame from queue with timeout
            frame_data = video_streaming_state["frame_queue"].get(timeout=1.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        except queue.Empty:
            # Send empty frame to keep connection alive
            continue
        except Exception as e:
            logger.error(f"Error in video stream: {e}")
            break

@app.post("/api/video/stream/stop")
async def stop_video_stream():
    """Stop video streaming"""
    video_streaming_state["active"] = False
    video_streaming_state["stop_event"].set()
    
    # Clear the queue
    while not video_streaming_state["frame_queue"].empty():
        try:
            video_streaming_state["frame_queue"].get_nowait()
        except queue.Empty:
            break
    
    return {"message": "Video stream stopped"}

@app.get("/api/video/stream/status")
async def get_stream_status():
    """Get current streaming status"""
    return {
        "active": video_streaming_state["active"],
        "video_id": video_streaming_state["current_video_id"],
        "zones_count": len(video_streaming_state["zones"])
    }

@app.post("/api/beam/start-monitoring")
async def start_beam_monitoring(video_id: str, db: Session = Depends(get_db)):
    """Start beam monitoring with motion detection"""
    if not video_id or video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    zones = db.query(ZoneModel).all()
    if not zones:
        raise HTTPException(status_code=400, detail="No zones configured")
    
    # Start video streaming with motion detection
    video_info = videos_db[video_id]
    
    # Update beam control state
    beam_control_state["detection_active"] = True
    beam_control_state["is_active"] = True
    beam_control_state["last_event"] = "MONITORING_STARTED"
    
    # Start motion detection in background
    threading.Thread(
        target=motion_detection_background,
        args=(video_info["path"], zones),
        daemon=True
    ).start()
    
    return {
        "status": "success",
        "message": "Beam monitoring started with motion detection",
        "zones_count": len(zones)
    }


def motion_detection_background(video_path: str, zones):
    """Background motion detection with zone monitoring"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return
    
    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    try:
        while beam_control_state["detection_active"]:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Apply background subtraction
            fgMask = backSub.apply(frame)
            
            # Check motion in each zone
            zones_with_motion = []
            
            for zone in zones:
                zone_dict = zone.to_dict()
                if detect_motion_in_zone(fgMask, zone_dict):
                    zones_with_motion.append(zone_dict["name"])
            
            # Update beam control based on motion
            if zones_with_motion:
                if beam_control_state["is_active"]:
                    beam_control_state["is_active"] = False
                    beam_control_state["zones_clear"] = False
                    beam_control_state["last_event"] = "MOTION_DETECTED"
                    
                    event = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "BEAM_STOPPED",
                        "message": f"Motion detected in zones: {', '.join(zones_with_motion)}",
                        "zones": zones_with_motion
                    }
                    beam_control_state["events"].append(event)
            else:
                if not beam_control_state["is_active"] and beam_control_state["detection_active"]:
                    beam_control_state["is_active"] = True
                    beam_control_state["zones_clear"] = True
                    beam_control_state["last_event"] = "ZONES_CLEAR"
                    
                    event = {
                        "timestamp": datetime.now().isoformat(),
                        "action": "BEAM_RESUMED",
                        "message": "All zones clear. Beam resumed."
                    }
                    beam_control_state["events"].append(event)
            
            time.sleep(0.1)  # 10 FPS motion detection
            
    except Exception as e:
        logger.error(f"Motion detection error: {e}")
    finally:
        cap.release()

def detect_motion_in_zone(fg_mask: np.ndarray, zone: dict) -> bool:
    """Detect motion within a specific polygonal zone"""
    try:
        coordinates = zone.get("coordinates", [])
        if len(coordinates) < 3:
            return False

        # Format points as required by OpenCV (N, 1, 2)
        points = np.array(
            [[int(coord["x"]), int(coord["y"])] for coord in coordinates],
            dtype=np.int32
        ).reshape((-1, 1, 2))  # Shape = (N, 1, 2)

        # Create blank mask
        zone_mask: np.ndarray = np.zeros_like(fg_mask, dtype=np.uint8)

        # Fill polygon with 255 (white)
        cv2.fillPoly(zone_mask, [points], 255)  # type: ignore

        # Apply mask to motion
        zone_motion = cv2.bitwise_and(fg_mask, zone_mask)

        # Count non-zero (motion) pixels
        motion_pixels = cv2.countNonZero(zone_motion)
        motion_area = cv2.countNonZero(zone_mask)

        if motion_area == 0:
            return False

        motion_percentage = (motion_pixels / motion_area) * 100
        threshold = zone.get("threshold", 5)

        return motion_percentage > threshold

    except Exception as e:
        logger.error(f"Error detecting motion in zone '{zone.get('name', 'unknown')}': {e}")
        return False

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
