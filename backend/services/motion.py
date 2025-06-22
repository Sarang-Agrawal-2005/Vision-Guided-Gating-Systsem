import cv2
import numpy as np

def create_zone_mask(frame_shape, coordinates):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(coordinates, np.int32)
    cv2.fillPoly(mask, [pts], (255,))
    return mask

def detect_motion_in_zones(frame, baseline_frame, zones):
    # Preprocess frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    baseline_gray = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2GRAY)
    baseline_gray = cv2.GaussianBlur(baseline_gray, (21, 21), 0)
    frame_diff = cv2.absdiff(baseline_gray, gray)
    
    active_zones = []
    zone_motions = {}
    
    for zone_name, zone_data in zones.items():
        mask = create_zone_mask(frame.shape, zone_data["coordinates"])
        masked_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask)
        thresh = cv2.threshold(masked_diff, zone_data["threshold"], 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > zone_data["min_area"] for c in contours)
        zone_motions[zone_name] = motion_detected
        if motion_detected:
            active_zones.append(zone_name)
    
    return zone_motions, active_zones

def draw_zones_on_frame_enhanced(frame, zones, active_zones=None, current_points=None):
    display_frame = frame.copy()
    priority_colors = {
        1: (255, 0, 0),    # Red
        2: (255, 128, 0),  # Orange
        3: (255, 255, 0),  # Yellow
        4: (0, 255, 0),    # Green
        5: (0, 0, 255)     # Blue
    }
    
    for zone_name, zone_data in zones.items():
        coordinates = zone_data.get('coordinates', [])
        if len(coordinates) >= 3:
            pts = np.array(coordinates, np.int32).reshape((-1, 1, 2))
            color = priority_colors.get(zone_data.get('priority', 1), (128, 128, 128))
            
            if active_zones and zone_name in active_zones:
                color = (0, 255, 255)  # Cyan
                thickness = 8
            else:
                thickness = 4
                
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0, display_frame)
            cv2.polylines(display_frame, [pts], True, color, thickness)
            
            # Add label
            if coordinates:
                label = f"{zone_name} (P{zone_data.get('priority', 1)})"
                label_pos = (coordinates[0][0], coordinates[0][1] - 10)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display_frame,
                              (label_pos[0] - 5, label_pos[1] - text_height - 5),
                              (label_pos[0] + text_width + 5, label_pos[1] + 5),
                              (0, 0, 0), -1)
                cv2.putText(display_frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
    
    return display_frame
