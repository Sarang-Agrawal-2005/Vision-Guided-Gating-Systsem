import streamlit as st
import cv2
import json
import numpy as np
from PIL import Image
import os
from datetime import timedelta
import time
from streamlit_image_coordinates import streamlit_image_coordinates

# Set page config
st.set_page_config(
    page_title="Motion Detection Beam Control System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def format_timestamp(seconds):
    """Format timestamp for display"""
    return str(timedelta(seconds=int(seconds)))

def load_video_frame(video_path):
    """Load the first frame from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
        return None

def save_zone_config(zones):
    """Save zones configuration to JSON file"""
    try:
        config = {"zones": zones}
        with open("zones_config.json", "w") as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def load_zone_config():
    """Load existing zone configuration"""
    try:
        if os.path.exists("zones_config.json"):
            with open("zones_config.json", "r") as f:
                return json.load(f)["zones"]
        return {}
    except:
        return {}

def create_zone_mask(frame_shape, coordinates):
    """Create mask for zone"""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(coordinates, np.int32)
    cv2.fillPoly(mask, [pts], (255,))
    return mask

def draw_zones_on_frame_enhanced(frame, zones, active_zones=None, current_points=None):
    """Draw zones on frame with enhanced visibility"""
    display_frame = frame.copy()
    
    # Colors for different priorities (more vibrant)
    priority_colors = {
        1: (255, 0, 0),      # Bright Red
        2: (255, 128, 0),    # Orange
        3: (255, 255, 0),    # Yellow
        4: (0, 255, 0),      # Green
        5: (0, 0, 255)       # Blue
    }
    
    # Draw existing zones
    for zone_name, zone_data in zones.items():
        coordinates = zone_data.get('coordinates', [])
        if len(coordinates) >= 3:
            # Convert coordinates to numpy array
            pts = np.array(coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))  # Reshape for cv2.polylines
            
            priority = zone_data.get('priority', 1)
            color = priority_colors.get(priority, (128, 128, 128))
            
            # Highlight active zones
            if active_zones and zone_name in active_zones:
                color = (0, 255, 255)  # Cyan for active motion
                thickness = 8
            else:
                thickness = 4
            
            # Draw filled polygon with transparency
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0, display_frame)
            
            # Draw zone boundary (thicker lines)
            cv2.polylines(display_frame, [pts], True, color, thickness)
            
            # Add zone label with background
            if len(coordinates) > 0:
                label = f"{zone_name} (P{priority})"
                label_pos = (coordinates[0][0], coordinates[0][1] - 10)
                
                # Draw text background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(
                    display_frame,
                    (label_pos[0] - 5, label_pos[1] - text_height - 5),
                    (label_pos[0] + text_width + 5, label_pos[1] + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    display_frame, 
                    label, 
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255),  # White text
                    2
                )
    
    # Draw current points being drawn
    if current_points and len(current_points) > 0:
        for i, point in enumerate(current_points):
            cv2.circle(display_frame, tuple(point), 8, (255, 255, 0), -1)
            cv2.putText(display_frame, str(i+1), 
                       (point[0]+15, point[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw lines between points
        if len(current_points) > 1:
            for i in range(len(current_points) - 1):
                cv2.line(display_frame, 
                        tuple(current_points[i]), 
                        tuple(current_points[i+1]), 
                        (255, 255, 0), 4)
        
        # Close polygon preview if 3+ points
        if len(current_points) >= 3:
            cv2.line(display_frame, 
                    tuple(current_points[-1]), 
                    tuple(current_points[0]), 
                    (255, 255, 0), 2)
    
    return display_frame

def detect_motion_in_zones(frame, baseline_frame, zones):
    """Detect motion in each zone"""
    # Preprocess frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    baseline_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
    baseline_gray = cv2.GaussianBlur(baseline_gray, (21, 21), 0)
    
    # Calculate frame difference
    frame_diff = cv2.absdiff(baseline_gray, gray)
    
    active_zones = []
    zone_motions = {}
    
    for zone_name, zone_data in zones.items():
        # Create zone mask
        mask = create_zone_mask(frame.shape, zone_data["coordinates"])
        
        # Apply mask to frame difference
        masked_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask)
        
        # Apply threshold
        thresh = cv2.threshold(masked_diff, zone_data["threshold"], 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > zone_data["min_area"] for c in contours)
        
        zone_motions[zone_name] = motion_detected
        if motion_detected:
            active_zones.append(zone_name)
    
    return zone_motions, active_zones

def create_video_with_zones_web_compatible(input_video_path, zones, output_path):
    """Create web-compatible video by using the SAME codec as the original video"""
    try:
        # Get original video properties
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.write(f"Video dimensions: {width}x{height}")
        st.write(f"FPS: {fps}")
        
        # SOLUTION: Copy the original video's codec by reading its fourcc
        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Convert fourcc to readable format
        fourcc_chars = [chr((original_fourcc >> 8 * i) & 0xFF) for i in range(4)]
        fourcc_str = ''.join(fourcc_chars)
        st.write(f"Original video codec: {fourcc_str}")
        
        # Use the same fourcc as the original video
        out = cv2.VideoWriter(output_path, original_fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.warning(f"Failed with original codec {fourcc_str}, trying H264...")
            # Fallback to H264 if available
            try:
                fourcc_h264 = cv2.VideoWriter.fourcc(*'H264')
                out = cv2.VideoWriter(output_path, fourcc_h264, fps, (width, height))
                if not out.isOpened():
                    raise Exception("H264 not available")
                st.info("Using H264 codec")
            except:
                st.warning("H264 not available, using mp4v...")
                fourcc_mp4v = cv2.VideoWriter.fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc_mp4v, fps, (width, height))
                if not out.isOpened():
                    st.error("Failed to open VideoWriter with any codec")
                    return False
                st.info("Using mp4v codec")
        else:
            st.success(f"Using original video codec: {fourcc_str}")
        
        baseline_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use first frame as baseline
            if baseline_frame is None:
                baseline_frame = frame_rgb.copy()
            
            # Detect motion and get active zones
            zone_motions, active_zones = detect_motion_in_zones(frame_rgb, baseline_frame, zones)
            
            # Draw zones on frame with enhanced visibility
            frame_with_zones = draw_zones_on_frame_enhanced(frame_rgb, zones, active_zones)
            
            # Convert back to BGR for video writing
            frame_bgr = cv2.cvtColor(frame_with_zones, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            frame_count += 1
            
            if frame_count > 1000:
                break
        
        cap.release()
        out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success(f"✅ Video processed successfully! Output size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
            return True
        else:
            st.error("❌ Failed to create output video file")
            return False
        
    except Exception as e:
        st.error(f"Error creating video with zones: {str(e)}")
        return False

def check_processed_video_exists():
    """Check if processed video exists and return status"""
    output_path = os.path.abspath("processed_video_with_zones.mp4")
    exists = os.path.exists(output_path)
    size = os.path.getsize(output_path) if exists else 0
    return exists, size, output_path

def main():
    st.title("🎯 Motion Detection Beam Control System")
    st.markdown("Complete system for zone setup and real-time beam control")
    
    # Initialize session state
    if 'zones' not in st.session_state:
        st.session_state.zones = load_zone_config()
    if 'video_frame' not in st.session_state:
        st.session_state.video_frame = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'baseline_frame' not in st.session_state:
        st.session_state.baseline_frame = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Zone Setup"
    if 'beam_active' not in st.session_state:
        st.session_state.beam_active = False
    if 'beam_events' not in st.session_state:
        st.session_state.beam_events = []
    if 'motion_detection_active' not in st.session_state:
        st.session_state.motion_detection_active = False
    if 'current_points' not in st.session_state:
        st.session_state.current_points = []
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎮 System Controls")
        
        # Mode selection
        st.session_state.current_mode = st.selectbox(
            "Select Mode",
            ["Zone Setup", "Beam Control"],
            index=["Zone Setup", "Beam Control"].index(st.session_state.current_mode)
        )
        
        st.divider()
        
        # Video upload
        st.subheader("📹 Video Upload")
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your test video"
        )
        
        if uploaded_file:
            st.session_state.video_path = "temp_video.mp4"
            with open(st.session_state.video_path, "wb") as f:
                f.write(uploaded_file.read())
            
            frame = load_video_frame(st.session_state.video_path)
            if frame is not None:
                st.session_state.video_frame = frame
                st.session_state.baseline_frame = frame.copy()
                st.success("✅ Video loaded successfully!")
        
        st.divider()
        
        # Mode-specific controls
        if st.session_state.current_mode == "Zone Setup":
            zone_setup_sidebar()
        elif st.session_state.current_mode == "Beam Control":
            beam_control_sidebar()
    
    # Main content based on mode
    if st.session_state.current_mode == "Zone Setup":
        zone_setup_main()
    elif st.session_state.current_mode == "Beam Control":
        beam_control_main()

def zone_setup_sidebar():
    """Zone setup sidebar controls with mouse drawing"""
    st.subheader("🖱️ Mouse Draw Zones")
    
    # Drawing controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Start Drawing"):
            st.session_state.drawing_mode = True
            st.session_state.current_points = []
            st.rerun()
    
    with col2:
        if st.button("🛑 Stop Drawing"):
            st.session_state.drawing_mode = False
            st.rerun()
    
    # Current drawing status
    if st.session_state.drawing_mode:
        st.success("🟢 Drawing Mode Active - Click on image to add points")
    else:
        st.info("⚪ Drawing Mode Inactive")
    
    # Current points
    if st.session_state.current_points:
        st.write(f"**Current Points:** {len(st.session_state.current_points)}")
        if len(st.session_state.current_points) >= 3:
            st.success("✅ Ready to create zone!")
    
    # Clear points
    if st.button("🧹 Clear Points"):
        st.session_state.current_points = []
        st.rerun()
    
    # Zone configuration form
    if len(st.session_state.current_points) >= 3:
        st.subheader("⚙️ Zone Configuration")
        with st.form("zone_form"):
            zone_name = st.text_input("Zone Name", placeholder="e.g., critical")
            
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Threshold", 10, 50, 25)
                min_area = st.slider("Min Area", 100, 2000, 500)
            with col2:
                motion_frames = st.slider("Motion Frames", 1, 10, 3)
                priority = st.slider("Priority", 1, 5, 1)
            
            if st.form_submit_button("💾 Save Zone"):
                if zone_name and zone_name not in st.session_state.zones:
                    st.session_state.zones[zone_name] = {
                        "coordinates": st.session_state.current_points.copy(),
                        "threshold": threshold,
                        "min_area": min_area,
                        "motion_frames": motion_frames,
                        "priority": priority
                    }
                    save_zone_config(st.session_state.zones)
                    st.success(f"✅ Zone '{zone_name}' created!")
                    st.session_state.current_points = []
                    st.session_state.drawing_mode = False
                    st.rerun()
                elif zone_name in st.session_state.zones:
                    st.error("❌ Zone name already exists")
                else:
                    st.error("❌ Please enter a zone name")
    
    # Existing zones
    if st.session_state.zones:
        st.subheader("📋 Existing Zones")
        for zone_name in list(st.session_state.zones.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                priority = st.session_state.zones[zone_name]['priority']
                st.write(f"🎯 {zone_name} (P{priority})")
            with col2:
                if st.button("🗑️", key=f"del_{zone_name}"):
                    del st.session_state.zones[zone_name]
                    save_zone_config(st.session_state.zones)
                    st.rerun()

def beam_control_sidebar():
    """Beam control sidebar controls"""
    st.subheader("⚡ Beam Control")
    
    if not st.session_state.zones:
        st.warning("⚠️ No zones configured!")
        st.info("Go to Zone Setup first to create detection zones.")
        return
    
    if not st.session_state.video_path:
        st.warning("⚠️ No video uploaded!")
        st.info("Please upload a video file first.")
        return
    
    # Check if processed video exists
    video_exists, video_size, video_path = check_processed_video_exists()
    
    # Show current status
    if video_exists and video_size > 0:
        st.success(f"✅ Processed video exists ({video_size / (1024*1024):.1f} MB)")
    else:
        st.info("ℹ️ No processed video found")
    
    # Show zone information
    st.subheader("📋 Zones to be embedded:")
    for zone_name, zone_data in st.session_state.zones.items():
        coords = zone_data.get('coordinates', [])
        st.write(f"**{zone_name}**: {len(coords)} points, Priority {zone_data['priority']}")
    
    # Process video with zones
    if st.button("🎬 Process Video with Zones (Web Compatible)"):
        if not st.session_state.zones:
            st.error("No zones to embed!")
            return
            
        with st.spinner("Creating web-compatible video with embedded zones..."):
            output_path = "processed_video_with_zones.mp4"
            
            # Remove existing processed video if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    st.info("Removed existing processed video")
                except:
                    pass
            
            success = create_video_with_zones_web_compatible(
                st.session_state.video_path, 
                st.session_state.zones, 
                output_path
            )
            
            if success:
                st.success("✅ Video processing completed!")
                st.rerun()
            else:
                st.error("❌ Failed to process video")
    
    # Beam status
    beam_status = "🟢 ACTIVE" if st.session_state.beam_active else "🔴 STOPPED"
    st.metric("Beam Status", beam_status)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Detection", disabled=st.session_state.motion_detection_active):
            st.session_state.motion_detection_active = True
            st.session_state.beam_active = True
            st.session_state.beam_events = []
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Detection", disabled=not st.session_state.motion_detection_active):
            st.session_state.motion_detection_active = False
            st.session_state.beam_active = False
            st.rerun()
    
    # Recent events
    if st.session_state.beam_events:
        st.subheader("📋 Recent Events")
        for event in st.session_state.beam_events[-5:]:
            action_icon = "🔴" if event['action'] == 'STOPPED' else "🟢"
            st.write(f"{action_icon} {event['message']}")

def zone_setup_main():
    """Zone setup main content with mouse drawing"""
    if st.session_state.video_frame is not None:
        st.subheader("🖱️ Mouse Draw Zone Boundaries")
        
        # Create display frame with zones and current drawing
        display_frame = draw_zones_on_frame_enhanced(
            st.session_state.video_frame, 
            st.session_state.zones, 
            current_points=st.session_state.current_points
        )
        
        # Convert to PIL Image for streamlit_image_coordinates
        pil_image = Image.fromarray(display_frame.astype('uint8'))
        
        # Interactive image with mouse coordinates
        if st.session_state.drawing_mode:
            st.info("🖱️ Click on the image to add points for your zone boundary")
            
            # Use streamlit_image_coordinates for mouse interaction
            coordinates = streamlit_image_coordinates(
                pil_image,
                key="image_coordinates"
            )
            
            # Add clicked point to current points
            if coordinates and st.session_state.drawing_mode:
                new_point = [coordinates["x"], coordinates["y"]]
                
                # Check if this is a new point (avoid duplicates from re-runs)
                if not st.session_state.current_points or new_point != st.session_state.current_points[-1]:
                    st.session_state.current_points.append(new_point)
                    st.success(f"Point added: ({coordinates['x']}, {coordinates['y']})")
                    st.rerun()
        else:
            st.image(pil_image, caption="Video frame with zones (click 'Start Drawing' to add new zones)", use_column_width=True)
        
        # Zone statistics
        if st.session_state.zones:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Zones", len(st.session_state.zones))
            with col2:
                priorities = [z['priority'] for z in st.session_state.zones.values()]
                st.metric("Highest Priority", min(priorities))
            with col3:
                avg_threshold = sum(z['threshold'] for z in st.session_state.zones.values()) / len(st.session_state.zones)
                st.metric("Avg Threshold", f"{avg_threshold:.1f}")
    else:
        st.info("👆 Please upload a video file to start zone configuration")

def beam_control_main():
    """Beam control main content with actual video playback"""
    if not st.session_state.zones:
        st.info("No zones configured. Please set up zones first.")
        return
    
    if not st.session_state.video_path:
        st.info("Please upload a video file first.")
        return
    
    st.subheader("⚡ Beam Control with Live Video Monitoring")
    
    # Video playback section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎬 Video Playback with Zones")
        
        # Check for processed video in real-time
        video_exists, video_size, video_path = check_processed_video_exists()
        
        if video_exists and video_size > 0:
            # Show processed video with zones embedded - SAME METHOD AS ORIGINAL
            st.success("🎯 Video with embedded zones (Same codec as original)")
            
            try:
                # Use the EXACT SAME method as original video
                st.video(video_path)  # Direct file path method
                
                st.info("✅ This video shows your zones overlaid on the original footage using the same codec")
                
            except Exception as e:
                st.error(f"Error loading processed video: {str(e)}")
                # Fallback to bytes method
                try:
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                except:
                    show_original_video()
        else:
            show_original_video()
    
    with col2:
        st.subheader("📊 Zone Information")
        
        # Display zone information
        for zone_name, zone_data in st.session_state.zones.items():
            priority_icons = ["🔴", "🟠", "🟡", "🟢", "🔵"]
            icon = priority_icons[min(zone_data['priority']-1, 4)]
            
            with st.expander(f"{icon} {zone_name}"):
                st.write(f"**Priority:** {zone_data['priority']}")
                st.write(f"**Threshold:** {zone_data['threshold']}")
                st.write(f"**Min Area:** {zone_data['min_area']}")
                st.write(f"**Motion Frames:** {zone_data['motion_frames']}")
        
        # Live simulation status
        st.subheader("🔴 Live Status")
        
        if st.session_state.motion_detection_active:
            # Simple simulation
            import random
            zones = list(st.session_state.zones.keys())
            
            if zones and random.random() < 0.3:
                active_zone = random.choice(zones)
                st.error(f"🚨 Motion detected in {active_zone}")
                
                if st.session_state.beam_active:
                    event = {
                        'timestamp': time.time(),
                        'action': 'STOPPED',
                        'zone': active_zone,
                        'message': f"Motion in {active_zone}. BEAM STOPPED."
                    }
                    st.session_state.beam_events.append(event)
                    st.session_state.beam_active = False
            else:
                st.success("✅ All zones clear")
                
                if not st.session_state.beam_active:
                    event = {
                        'timestamp': time.time(),
                        'action': 'RESUMED',
                        'zone': 'all_clear',
                        'message': "All zones clear. BEAM RESUMED."
                    }
                    st.session_state.beam_events.append(event)
                    st.session_state.beam_active = True
            
            # Auto-refresh for simulation
            time.sleep(2)
            st.rerun()
        else:
            st.info("Motion detection stopped")

def show_original_video():
    """Helper function to show original video"""
    st.warning("📹 Original video (click 'Process Video with Zones' to see zones embedded)")
    
    try:
        # Use direct file path method - same as processed video will use
        st.video(st.session_state.video_path)
        
        st.info("⚠️ Click 'Process Video with Zones' in the sidebar to see zones embedded in the video")
        
    except Exception as e:
        st.error(f"Error loading original video: {str(e)}")

if __name__ == "__main__":
    main()
