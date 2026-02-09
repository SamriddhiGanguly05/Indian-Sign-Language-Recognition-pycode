"""
Gesture-Based Sign Language Recognition with Buffer Time
Uses brute force gesture detection (thumbs up, palm up) instead of ML model
5-second buffer before recognition starts
"""

import cv2
import numpy as np
import time
import os
import mediapipe as mp
from collections import deque

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

def detect_cheek_gesture(results):
    """Detect if index finger is on right cheek using face and hand landmarks"""
    if not results.face_landmarks or not results.right_hand_landmarks:
        return False, 0.0
    
    face_landmarks = results.face_landmarks.landmark
    right_hand = results.right_hand_landmarks.landmark
    
    # Get right cheek region from face landmarks
    cheek_landmarks = [
        face_landmarks[234],  # Right cheek
        face_landmarks[454],  # Right cheek area
        face_landmarks[264],  # Right cheekbone
    ]
    
    # Calculate cheek center (average position)
    cheek_center_x = sum([lm.x for lm in cheek_landmarks]) / len(cheek_landmarks)
    cheek_center_y = sum([lm.y for lm in cheek_landmarks]) / len(cheek_landmarks)
    
    # Get right index finger tip (landmark 8)
    index_tip = right_hand[8]
    
    # Check if index finger is near cheek (within threshold distance)
    distance_x = abs(index_tip.x - cheek_center_x)
    distance_y = abs(index_tip.y - cheek_center_y)
    
    # Threshold for "near" cheek (adjustable)
    threshold_x = 0.08  # ~8% of image width
    threshold_y = 0.08  # ~8% of image height
    
    # Check if index finger is extended (tip is above MCP)
    index_extended = index_tip.y < right_hand[5].y
    
    # Check if index finger is near cheek and extended
    is_near_cheek = (distance_x < threshold_x and distance_y < threshold_y and index_extended)
    
    # Calculate confidence based on distance (closer = higher confidence)
    if is_near_cheek:
        # Normalize distance to confidence (0-1)
        max_distance = (threshold_x + threshold_y) / 2
        actual_distance = (distance_x + distance_y) / 2
        confidence = 1.0 - (actual_distance / max_distance)
        confidence = max(0.5, min(1.0, confidence))  # Clamp between 0.5 and 1.0
        return True, confidence * 100
    else:
        return False, 0.0

def detect_both_palms_forward(results):
    """Detect if both palms are facing forward/outward"""
    if not results.left_hand_landmarks or not results.right_hand_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark
    right_hand = results.right_hand_landmarks.landmark
    
    # Check if both palms are facing forward (fingers extended, palm visible)
    # Palm forward: wrist is behind fingers (higher z or lower y for wrist)
    # Fingers extended: tips are in front of MCPs
    
    # Left hand checks
    left_wrist_behind = left_hand[0].z > left_hand[9].z  # Wrist behind middle MCP
    left_fingers_extended = (left_hand[8].y < left_hand[5].y and  # Index extended
                            left_hand[12].y < left_hand[9].y and  # Middle extended
                            left_hand[16].y < left_hand[13].y)     # Ring extended
    
    # Right hand checks
    right_wrist_behind = right_hand[0].z > right_hand[9].z
    right_fingers_extended = (right_hand[8].y < right_hand[5].y and
                              right_hand[12].y < right_hand[9].y and
                              right_hand[16].y < right_hand[13].y)
    
    # Both palms forward
    both_palms_forward = (left_wrist_behind and left_fingers_extended and
                         right_wrist_behind and right_fingers_extended)
    
    if both_palms_forward:
        # Calculate confidence based on how well conditions are met
        conditions = [left_wrist_behind, left_fingers_extended, 
                     right_wrist_behind, right_fingers_extended]
        confidence = sum(conditions) / len(conditions)
        return True, confidence * 100
    
    return False, 0.0

def detect_hands_intersection(results):
    """Detect if topmost points of blue (left) and green (right) hands intersect/touch"""
    if not results.left_hand_landmarks or not results.right_hand_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark  # Blue hand
    right_hand = results.right_hand_landmarks.landmark  # Green hand
    
    # Find the topmost point of each hand (minimum y value, since y=0 is top)
    # Check all landmarks to find the one with smallest y coordinate
    left_topmost = left_hand[0]  # Start with first landmark
    left_topmost_y = left_hand[0].y
    
    for landmark in left_hand:
        if landmark.y < left_topmost_y:
            left_topmost = landmark
            left_topmost_y = landmark.y
    
    right_topmost = right_hand[0]  # Start with first landmark
    right_topmost_y = right_hand[0].y
    
    for landmark in right_hand:
        if landmark.y < right_topmost_y:
            right_topmost = landmark
            right_topmost_y = landmark.y
    
    # Calculate distance between the two topmost points
    distance_x = abs(left_topmost.x - right_topmost.x)
    distance_y = abs(left_topmost.y - right_topmost.y)
    distance = (distance_x ** 2 + distance_y ** 2) ** 0.5
    
    # Threshold for intersection (distance in normalized coordinates)
    intersection_threshold = 0.08  # ~8% of image size
    
    # Check if topmost points are intersecting
    if distance < intersection_threshold:
        # Calculate confidence based on distance (closer = higher confidence)
        confidence = min(95.0, 70.0 + (25.0 * (1.0 - distance / intersection_threshold)))
        return True, confidence
    
    return False, 0.0

def detect_india_gesture(results):
    """Detect India gesture: Blue hand thumb above eyebrows AND blue hand above shoulder baseline"""
    if not results.left_hand_landmarks or not results.pose_landmarks or not results.face_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark  # Blue hand
    pose_landmarks = results.pose_landmarks.landmark
    face_landmarks = results.face_landmarks.landmark
    
    # Get shoulder baseline (average y-coordinate of left and right shoulders)
    # MediaPipe pose landmarks: 11 = left shoulder, 12 = right shoulder
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    shoulder_baseline_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Get eyebrow level (average y-coordinate of eyebrow landmarks)
    # MediaPipe face landmarks for eyebrows:
    # Left eyebrow: 107 (outer), 66 (middle), 69 (inner)
    # Right eyebrow: 336 (outer), 296 (middle), 299 (inner)
    eyebrow_landmarks = [
        face_landmarks[107],  # Left eyebrow outer
        face_landmarks[66],   # Left eyebrow middle
        face_landmarks[69],   # Left eyebrow inner
        face_landmarks[336], # Right eyebrow outer
        face_landmarks[296],  # Right eyebrow middle
        face_landmarks[299]   # Right eyebrow inner
    ]
    eyebrow_y_coords = [lm.y for lm in eyebrow_landmarks]
    eyebrow_baseline_y = sum(eyebrow_y_coords) / len(eyebrow_y_coords)
    
    # Get blue hand thumb (landmark 4 is thumb tip)
    blue_hand_thumb = left_hand[4]  # Thumb tip
    thumb_y = blue_hand_thumb.y
    
    # Condition 1: Blue hand thumb must be above eyebrows
    thumb_above_eyebrows = thumb_y < eyebrow_baseline_y
    
    if not thumb_above_eyebrows:
        return False, 0.0
    
    # Condition 2: Blue hand must be above shoulder baseline
    # Find the topmost point of the left hand
    left_hand_topmost = left_hand[0]
    left_hand_topmost_y = left_hand[0].y
    
    for landmark in left_hand:
        if landmark.y < left_hand_topmost_y:
            left_hand_topmost = landmark
            left_hand_topmost_y = landmark.y
    
    hand_above_shoulder = left_hand_topmost_y < shoulder_baseline_y
    
    if not hand_above_shoulder:
        return False, 0.0
    
    # Both conditions must be met
    if thumb_above_eyebrows and hand_above_shoulder:
        # Calculate confidence based on how well conditions are met
        # Distance of thumb above eyebrows
        distance_above_eyebrows = eyebrow_baseline_y - thumb_y
        eyebrow_confidence = min(1.0, distance_above_eyebrows / 0.05)  # Normalize by reasonable distance
        
        # Distance of hand above shoulder
        distance_above_shoulder = shoulder_baseline_y - left_hand_topmost_y
        shoulder_confidence = min(1.0, distance_above_shoulder / 0.1)  # Normalize by reasonable distance
        
        confidence = 70.0 + (15.0 * eyebrow_confidence) + (15.0 * shoulder_confidence)
        confidence = min(95.0, confidence)
        return True, confidence
    
    return False, 0.0

def detect_female_gesture(results):
    """Detect Female gesture: Blue hand TOUCHING the nose"""
    if not results.left_hand_landmarks or not results.face_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark  # Blue hand (dominant hand)
    face_landmarks = results.face_landmarks.landmark
    
    # Get nose tip (landmark 4) - the main point to touch
    nose_tip = face_landmarks[4]  # Nose tip
    
    # Focus on fingertips that can actually touch the nose
    # Primary: Index finger tip (most common for pointing/touching)
    # Secondary: Thumb tip, Middle finger tip
    hand_touch_points = [
        left_hand[8],   # Index tip (primary)
        left_hand[4],   # Thumb tip
        left_hand[12],  # Middle tip
    ]
    
    # Strict threshold for TOUCHING (much smaller - actual contact)
    touch_threshold = 0.04  # ~4% of image size - requires actual touching
    
    # Check if any fingertip is TOUCHING the nose
    min_distance = float('inf')
    touch_count = 0
    
    for touch_point in hand_touch_points:
        # Calculate distance to nose tip
        distance_x = abs(touch_point.x - nose_tip.x)
        distance_y = abs(touch_point.y - nose_tip.y)
        distance = (distance_x ** 2 + distance_y ** 2) ** 0.5
        
        min_distance = min(min_distance, distance)
        
        if distance < touch_threshold:
            touch_count += 1
    
    # Only detect if hand is actually TOUCHING the nose
    if touch_count >= 1:  # At least one fingertip is touching nose
        # Calculate confidence based on how close the touch is
        confidence = min(95.0, 75.0 + (20.0 * (1.0 - min_distance / touch_threshold)))
        return True, confidence
    
    return False, 0.0

def detect_male_gesture(results):
    """Detect Male gesture: Blue hand TOUCHING the lips"""
    if not results.left_hand_landmarks or not results.face_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark  # Blue hand (dominant hand)
    face_landmarks = results.face_landmarks.landmark
    
    # Get lip/mouth center - use actual lip landmarks if available
    nose_tip = face_landmarks[4]  # Nose tip
    
    # Estimate mouth position (below nose)
    lip_center_x = nose_tip.x
    lip_center_y = nose_tip.y + 0.04  # Mouth is below nose
    
    # Try to get actual mouth landmarks if available (MediaPipe face mesh has 468 points)
    # Upper lip: landmark 13, Lower lip: landmark 14
    if len(face_landmarks) > 14:
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        lip_center_x = (upper_lip.x + lower_lip.x) / 2
        lip_center_y = (upper_lip.y + lower_lip.y) / 2
    
    # Focus on fingertips that can actually touch the lips
    # Primary: Index finger tip (most common for pointing/touching)
    # Secondary: Thumb tip, Middle finger tip
    hand_touch_points = [
        left_hand[8],   # Index tip (primary)
        left_hand[4],   # Thumb tip
        left_hand[12],  # Middle tip
    ]
    
    # Strict threshold for TOUCHING (much smaller - actual contact)
    touch_threshold = 0.04  # ~4% of image size - requires actual touching
    
    # Check if any fingertip is TOUCHING the lips
    min_distance = float('inf')
    touch_count = 0
    
    for touch_point in hand_touch_points:
        # Calculate distance to lip center
        distance_x = abs(touch_point.x - lip_center_x)
        distance_y = abs(touch_point.y - lip_center_y)
        distance = (distance_x ** 2 + distance_y ** 2) ** 0.5
        
        min_distance = min(min_distance, distance)
        
        if distance < touch_threshold:
            touch_count += 1
    
    # Only detect if hand is actually TOUCHING the lips
    if touch_count >= 1:  # At least one fingertip is touching lips
        # Calculate confidence based on how close the touch is
        confidence = min(95.0, 75.0 + (20.0 * (1.0 - min_distance / touch_threshold)))
        return True, confidence
    
    return False, 0.0

def detect_thank_you_gesture(results):
    """Detect Thank you gesture: Blue hand TOUCHING the chin"""
    if not results.left_hand_landmarks or not results.face_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark  # Blue hand (dominant hand)
    face_landmarks = results.face_landmarks.landmark
    
    # Get chin position - chin is at the lower part of the face
    # MediaPipe face mesh: chin tip is typically around landmark 175 or 199
    # We can also estimate chin position from nose tip (chin is below nose)
    nose_tip = face_landmarks[4]  # Nose tip
    
    # Estimate chin position (below nose, further down)
    chin_x = nose_tip.x
    chin_y = nose_tip.y + 0.08  # Chin is below nose (about 8% of face height)
    
    # Try to get actual chin landmark if available (MediaPipe face mesh has 468 points)
    # Chin tip is typically at landmark 175 or 199
    if len(face_landmarks) > 175:
        chin_landmark = face_landmarks[175]  # Chin tip
        chin_x = chin_landmark.x
        chin_y = chin_landmark.y
    
    # Focus on fingertips that can actually touch the chin
    # Primary: Index finger tip (most common for pointing/touching)
    # Secondary: Thumb tip, Middle finger tip
    hand_touch_points = [
        left_hand[8],   # Index tip (primary)
        left_hand[4],   # Thumb tip
        left_hand[12],  # Middle tip
    ]
    
    # Strict threshold for TOUCHING (much smaller - actual contact)
    touch_threshold = 0.04  # ~4% of image size - requires actual touching
    
    # Check if any fingertip is TOUCHING the chin
    min_distance = float('inf')
    touch_count = 0
    
    for touch_point in hand_touch_points:
        # Calculate distance to chin
        distance_x = abs(touch_point.x - chin_x)
        distance_y = abs(touch_point.y - chin_y)
        distance = (distance_x ** 2 + distance_y ** 2) ** 0.5
        
        min_distance = min(min_distance, distance)
        
        if distance < touch_threshold:
            touch_count += 1
    
    # Only detect if hand is actually TOUCHING the chin
    if touch_count >= 1:  # At least one fingertip is touching chin
        # Calculate confidence based on how close the touch is
        confidence = min(95.0, 75.0 + (20.0 * (1.0 - min_distance / touch_threshold)))
        return True, confidence
    
    return False, 0.0

def detect_namaste_gesture(results):
    """
    Rule-based gesture recognizer for 'Namaste' / 'Prayer Hands' gesture.
    Uses MediaPipe landmarks with shoulder-width normalization.
    
    Returns: (bool, float) - (detected, confidence)
    """
    # Check if all required landmarks are available
    if not results.pose_landmarks:
        return False, 0.0
    if not results.left_hand_landmarks or not results.right_hand_landmarks:
        return False, 0.0
    if not results.face_landmarks:
        return False, 0.0
    
    # Get pose landmarks
    pose = results.pose_landmarks.landmark
    
    # MediaPipe pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    # Get shoulder positions
    left_shoulder = pose[LEFT_SHOULDER]
    right_shoulder = pose[RIGHT_SHOULDER]
    
    # Rule 1: Shoulder Width Normalization
    shoulder_width = np.sqrt(
        (right_shoulder.x - left_shoulder.x) ** 2 + 
        (right_shoulder.y - left_shoulder.y) ** 2
    )
    
    if shoulder_width < 0.05:  # Too small, likely not detected properly
        return False, 0.0
    
    # Get hand landmarks
    left_hand = results.left_hand_landmarks.landmark
    right_hand = results.right_hand_landmarks.landmark
    
    # Hand landmark indices
    WRIST = 0
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    # Get key points
    left_wrist = left_hand[WRIST]
    right_wrist = right_hand[WRIST]
    left_index_mcp = left_hand[INDEX_MCP]
    right_index_mcp = right_hand[INDEX_MCP]
    left_index_tip = left_hand[INDEX_TIP]
    right_index_tip = right_hand[INDEX_TIP]
    
    # Get pose wrist positions (for forearm angle calculation)
    pose_left_wrist = pose[LEFT_WRIST]
    pose_right_wrist = pose[RIGHT_WRIST]
    pose_left_elbow = pose[LEFT_ELBOW]
    pose_right_elbow = pose[RIGHT_ELBOW]
    
    # Get nose for angle calculation
    face_landmarks = results.face_landmarks.landmark
    nose_tip = face_landmarks[4]  # Nose tip
    
    # Helper function to calculate distance
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    # Helper function to calculate angle between three points (in degrees)
    def calculate_angle(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        # Vector from p2 to p1
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        # Vector from p2 to p3
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 180.0  # Invalid angle
        
        cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    # Rule 2: Wrist Distance Rule
    wrist_distance = distance(left_wrist, right_wrist)
    wrist_threshold = 0.20 * shoulder_width
    rule2_pass = wrist_distance < wrist_threshold
    
    # Rule 3: Palm Center Proximity Rule
    palm_distance = distance(left_index_mcp, right_index_mcp)
    palm_threshold = 0.18 * shoulder_width
    rule3_pass = palm_distance < palm_threshold
    
    # Rule 4: Finger Symmetry Rule
    # Check y-coordinates of fingertips (vertical alignment)
    y_diff = abs(left_index_tip.y - right_index_tip.y)
    y_threshold = 0.10 * shoulder_width
    y_symmetry = y_diff < y_threshold
    
    # Check x-coordinates overlap (horizontal alignment)
    x_diff = abs(left_index_tip.x - right_index_tip.x)
    x_threshold = 0.12 * shoulder_width
    x_symmetry = x_diff < x_threshold
    
    rule4_pass = y_symmetry and x_symmetry
    
    # Rule 5: Forearm Angle Rule
    # Calculate forearm angles toward nose
    left_forearm_angle = calculate_angle(pose_left_elbow, pose_left_wrist, nose_tip)
    right_forearm_angle = calculate_angle(pose_right_elbow, pose_right_wrist, nose_tip)
    
    angle_difference = abs(left_forearm_angle - right_forearm_angle)
    rule5_pass = angle_difference < 15.0
    
    # Rule 6: Wrist Height Rule
    wrist_height_diff = abs(left_wrist.y - right_wrist.y)
    height_threshold = 0.08 * shoulder_width
    rule6_pass = wrist_height_diff < height_threshold
    
    # Rule 7: Hand Orientation Rule
    # Fingertips must be above wrists (hands upright)
    left_hand_upright = left_index_tip.y < left_wrist.y
    right_hand_upright = right_index_tip.y < right_wrist.y
    rule7_pass = left_hand_upright and right_hand_upright
    
    # Check if ALL rules pass
    all_rules_pass = (
        rule2_pass and rule3_pass and rule4_pass and 
        rule5_pass and rule6_pass and rule7_pass
    )
    
    if all_rules_pass:
        # Calculate confidence based on how well each rule is satisfied
        # Closer to thresholds = higher confidence
        confidence_scores = []
        
        # Rule 2 confidence (wrist distance - closer is better)
        if wrist_distance > 0:
            rule2_conf = min(100.0, 100.0 * (1.0 - wrist_distance / wrist_threshold))
        else:
            rule2_conf = 100.0
        confidence_scores.append(rule2_conf)
        
        # Rule 3 confidence (palm distance - closer is better)
        if palm_distance > 0:
            rule3_conf = min(100.0, 100.0 * (1.0 - palm_distance / palm_threshold))
        else:
            rule3_conf = 100.0
        confidence_scores.append(rule3_conf)
        
        # Rule 4 confidence (symmetry - better alignment = higher)
        y_conf = min(100.0, 100.0 * (1.0 - y_diff / y_threshold)) if y_threshold > 0 else 100.0
        x_conf = min(100.0, 100.0 * (1.0 - x_diff / x_threshold)) if x_threshold > 0 else 100.0
        rule4_conf = (y_conf + x_conf) / 2.0
        confidence_scores.append(rule4_conf)
        
        # Rule 5 confidence (angle difference - smaller is better)
        rule5_conf = min(100.0, 100.0 * (1.0 - angle_difference / 15.0))
        confidence_scores.append(rule5_conf)
        
        # Rule 6 confidence (height difference - smaller is better)
        rule6_conf = min(100.0, 100.0 * (1.0 - wrist_height_diff / height_threshold)) if height_threshold > 0 else 100.0
        confidence_scores.append(rule6_conf)
        
        # Rule 7 is binary (pass/fail), so give it full confidence if passed
        confidence_scores.append(100.0)
        
        # Average confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return True, min(95.0, overall_confidence)
    
    return False, 0.0

def detect_sibling_gesture(results):
    """
    Detect Sibling (Brother) gesture: Dominant hand's middle finger touches same-side shoulder.
    - If right hand (green) is dominant → middle finger touches RIGHT shoulder (orange landmark)
    - If left hand (blue) is dominant → middle finger touches LEFT shoulder (orange landmark)
    """
    if not results.pose_landmarks:
        return False, 0.0
    
    # MediaPipe pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    # MediaPipe hand landmark indices
    MIDDLE_TIP = 12  # Middle finger tip
    
    pose = results.pose_landmarks.landmark
    left_shoulder = pose[LEFT_SHOULDER]
    right_shoulder = pose[RIGHT_SHOULDER]
    
    # Helper function to calculate distance
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    # Check right hand (green) middle finger touching RIGHT shoulder
    if results.right_hand_landmarks:
        right_hand = results.right_hand_landmarks.landmark
        right_middle_tip = right_hand[MIDDLE_TIP]
        
        # Check if right middle finger is touching right shoulder
        dist = distance(right_middle_tip, right_shoulder)
        touch_threshold = 0.04  # ~4% of image size - requires actual touching
        
        if dist < touch_threshold:
            # Calculate confidence based on how close the touch is
            confidence = min(95.0, 75.0 + (20.0 * (1.0 - dist / touch_threshold)))
            return True, confidence
    
    # Check left hand (blue) middle finger touching LEFT shoulder
    if results.left_hand_landmarks:
        left_hand = results.left_hand_landmarks.landmark
        left_middle_tip = left_hand[MIDDLE_TIP]
        
        # Check if left middle finger is touching left shoulder
        dist = distance(left_middle_tip, left_shoulder)
        touch_threshold = 0.04  # ~4% of image size - requires actual touching
        
        if dist < touch_threshold:
            # Calculate confidence based on how close the touch is
            confidence = min(95.0, 75.0 + (20.0 * (1.0 - dist / touch_threshold)))
            return True, confidence
    
    return False, 0.0

def detect_doctor_gesture(results):
    """
    Detect Doctor gesture: Hardcore brute force - Blue hand touches anywhere in shoulder width region.
    - Blue hand (blue dot) touches shoulder width region = Doctor detected
    """
    if not results.left_hand_landmarks or not results.pose_landmarks:
        return False, 0.0
    
    left_hand = results.left_hand_landmarks.landmark  # Blue hand (dominant hand)
    pose_landmarks = results.pose_landmarks.landmark
    
    # Get shoulder landmarks
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    left_shoulder = pose_landmarks[LEFT_SHOULDER]
    right_shoulder = pose_landmarks[RIGHT_SHOULDER]
    
    # Calculate shoulder width (distance between left and right shoulders)
    shoulder_width = np.sqrt(
        (right_shoulder.x - left_shoulder.x) ** 2 + 
        (right_shoulder.y - left_shoulder.y) ** 2
    )
    
    if shoulder_width < 0.05:  # Too small, likely not detected properly
        return False, 0.0
    
    # Calculate shoulder baseline (average y-coordinate)
    shoulder_baseline_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Get blue hand's fingertips that can touch
    blue_thumb_tip = left_hand[4]   # Thumb tip
    blue_index_tip = left_hand[8]   # Index tip
    blue_middle_tip = left_hand[12] # Middle tip
    blue_wrist = left_hand[0]       # Wrist
    
    # Helper function to calculate distance
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    
    # Check if blue hand is within shoulder width region
    # Define shoulder width region: between left and right shoulders, with some vertical tolerance
    shoulder_region_threshold = 0.15 * shoulder_width  # 15% of shoulder width for touch detection
    vertical_tolerance = 0.10 * shoulder_width  # 10% vertical tolerance
    
    # Check if any blue hand point is touching the shoulder width region
    min_distance = float('inf')
    touch_detected = False
    
    # Check all hand points against both shoulders
    hand_points = [blue_thumb_tip, blue_index_tip, blue_middle_tip, blue_wrist]
    
    for hand_point in hand_points:
        # Distance to left shoulder
        dist_to_left = distance(hand_point, left_shoulder)
        # Distance to right shoulder
        dist_to_right = distance(hand_point, right_shoulder)
        # Check if within shoulder width region (between shoulders or near them)
        # Also check if at similar height (within vertical tolerance)
        height_diff = abs(hand_point.y - shoulder_baseline_y)
        
        # Check if point is within shoulder width region
        # Method 1: Close to either shoulder landmark
        if dist_to_left < shoulder_region_threshold or dist_to_right < shoulder_region_threshold:
            if height_diff < vertical_tolerance:
                min_distance = min(min_distance, min(dist_to_left, dist_to_right))
                touch_detected = True
        
        # Method 2: Point is between shoulders horizontally and at shoulder height
        if (min(left_shoulder.x, right_shoulder.x) <= hand_point.x <= max(left_shoulder.x, right_shoulder.x)):
            if height_diff < vertical_tolerance:
                # Calculate distance to shoulder line
                dist_to_shoulder_line = height_diff
                min_distance = min(min_distance, dist_to_shoulder_line)
                touch_detected = True
    
    if touch_detected:
        # Calculate confidence based on how close the touch is
        confidence = min(95.0, 75.0 + (20.0 * (1.0 - min_distance / shoulder_region_threshold)))
        return True, confidence
    
    return False, 0.0

def detect_water_gesture(results):
    """
    Detect Water gesture: Green hand (right hand) goes inside the mouth hole for a long time.
    - Green hand (right hand) inside mouth region = "i want water"
    """
    if not results.right_hand_landmarks or not results.face_landmarks:
        return False, 0.0
    
    right_hand = results.right_hand_landmarks.landmark  # Green hand
    face_landmarks = results.face_landmarks.landmark
    
    # Get mouth landmarks - MediaPipe face mesh has 468 points
    # Mouth landmarks: upper lip, lower lip, mouth corners
    # Common mouth landmarks:
    # Upper lip center: around 13
    # Lower lip center: around 14
    # Mouth corners: around 61 (left) and 291 (right) or 78 (left) and 308 (right)
    
    nose_tip = face_landmarks[4]  # Nose tip for reference
    
    # Estimate mouth region if specific landmarks not available
    # Mouth is below nose
    mouth_center_x = nose_tip.x
    mouth_center_y = nose_tip.y + 0.05  # Mouth is below nose
    
    # Try to get actual mouth landmarks
    upper_lip_y = mouth_center_y - 0.02
    lower_lip_y = mouth_center_y + 0.02
    mouth_left_x = mouth_center_x - 0.03
    mouth_right_x = mouth_center_x + 0.03
    
    # Try to get actual mouth landmarks if available
    if len(face_landmarks) > 14:
        # Try to find mouth landmarks
        # Upper lip: landmark 13
        # Lower lip: landmark 14
        if len(face_landmarks) > 13:
            upper_lip = face_landmarks[13]
            upper_lip_y = upper_lip.y
            mouth_center_x = upper_lip.x
        
        if len(face_landmarks) > 14:
            lower_lip = face_landmarks[14]
            lower_lip_y = lower_lip.y
            mouth_center_x = (mouth_center_x + lower_lip.x) / 2
        
        # Try to get mouth corners for width
        # Left corner: around 61 or 78
        # Right corner: around 291 or 308
        if len(face_landmarks) > 78:
            mouth_left = face_landmarks[78]  # Left mouth corner
            mouth_left_x = mouth_left.x
        if len(face_landmarks) > 308:
            mouth_right = face_landmarks[308]  # Right mouth corner
            mouth_right_x = mouth_right.x
    
    # Get green hand points (fingertips and wrist)
    green_hand_points = [
        right_hand[4],   # Thumb tip
        right_hand[8],   # Index tip
        right_hand[12],  # Middle tip
        right_hand[16],  # Ring tip
        right_hand[20],  # Pinky tip
        right_hand[0],   # Wrist
    ]
    
    # Check if any green hand point is inside the mouth region
    # Inside mouth means:
    # 1. X coordinate is between mouth_left_x and mouth_right_x
    # 2. Y coordinate is between upper_lip_y and lower_lip_y
    # 3. Or very close to mouth center
    
    points_inside_mouth = 0
    min_distance_to_mouth = float('inf')
    
    for hand_point in green_hand_points:
        # Check if point is horizontally within mouth width
        x_inside = mouth_left_x <= hand_point.x <= mouth_right_x
        
        # Check if point is vertically within mouth height (between upper and lower lip)
        y_inside = upper_lip_y <= hand_point.y <= lower_lip_y
        
        # Also check if point is very close to mouth center (within threshold)
        distance_to_center = np.sqrt(
            (hand_point.x - mouth_center_x) ** 2 + 
            (hand_point.y - mouth_center_y) ** 2
        )
        close_to_center = distance_to_center < 0.05  # 5% of image size
        
        # Point is inside mouth if it's within mouth boundaries OR very close to center
        if (x_inside and y_inside) or close_to_center:
            points_inside_mouth += 1
            min_distance_to_mouth = min(min_distance_to_mouth, distance_to_center)
    
    # Require at least 1 point inside mouth for detection
    if points_inside_mouth >= 1:
        # Calculate confidence based on how many points are inside and how close
        confidence = min(95.0, 70.0 + (25.0 * (points_inside_mouth / len(green_hand_points))))
        if min_distance_to_mouth < 0.05:
            confidence = min(95.0, confidence + 10.0)
        return True, confidence
    
    return False, 0.0

def detect_head_gesture(results, head_history):
    """Detect head gestures: nod (up/down) = Yes, rotate (left/right) = No"""
    if not results.face_landmarks:
        return None, 0.0
    
    face_landmarks = results.face_landmarks.landmark
    
    # Use nose tip (landmark 4) and forehead (landmark 10) for head position
    nose_tip = face_landmarks[4]  # Nose tip
    forehead = face_landmarks[10]  # Forehead center
    
    # Calculate head center position
    head_center_x = (nose_tip.x + forehead.x) / 2
    head_center_y = (nose_tip.y + forehead.y) / 2
    
    # Add to history (keep last 10 frames for movement detection)
    head_history.append((head_center_x, head_center_y))
    if len(head_history) > 10:
        head_history.popleft()
    
    # Need at least 5 frames to detect movement
    if len(head_history) < 5:
        return None, 0.0
    
    # Calculate movement patterns
    recent_positions = list(head_history)
    
    # Calculate vertical movement (for nod detection)
    y_positions = [pos[1] for pos in recent_positions]
    y_range = max(y_positions) - min(y_positions)
    y_movement = abs(recent_positions[-1][1] - recent_positions[0][1])
    
    # Calculate horizontal movement (for rotate detection)
    x_positions = [pos[0] for pos in recent_positions]
    x_range = max(x_positions) - min(x_positions)
    x_movement = abs(recent_positions[-1][0] - recent_positions[0][0])
    
    # Thresholds for gesture detection
    nod_threshold = 0.03  # Vertical movement threshold
    rotate_threshold = 0.04  # Horizontal movement threshold
    
    # Detect head nod (Yes) - vertical movement dominates
    if y_range > nod_threshold and y_range > x_range * 1.5:
        # Check if there's an up-down pattern
        y_mean = sum(y_positions) / len(y_positions)
        y_variance = sum((y - y_mean) ** 2 for y in y_positions) / len(y_positions)
        if y_variance > 0.0001:  # Significant vertical variance
            confidence = min(90.0, (y_range / nod_threshold) * 30)
            return "head_nod", confidence
    
    # Detect head rotate (No) - horizontal movement dominates
    if x_range > rotate_threshold and x_range > y_range * 1.5:
        # Check if there's a left-right pattern
        x_mean = sum(x_positions) / len(x_positions)
        x_variance = sum((x - x_mean) ** 2 for x in x_positions) / len(x_positions)
        if x_variance > 0.0001:  # Significant horizontal variance
            confidence = min(90.0, (x_range / rotate_threshold) * 30)
            return "head_rotate", confidence
    
    return None, 0.0

def detect_gesture(results):
    """Detect specific gestures from MediaPipe results with all gestures"""
    # Check which hands are detected
    left_hand_detected = results.left_hand_landmarks is not None
    right_hand_detected = results.right_hand_landmarks is not None
    
    # First check for Namaste gesture (Prayer Hands) - highest priority
    namaste_detected, namaste_confidence = detect_namaste_gesture(results)
    if namaste_detected and namaste_confidence > 50.0:
        return "namaste", namaste_confidence
    
    # Check for hands intersection (Home) - high priority
    hands_intersect, intersect_confidence = detect_hands_intersection(results)
    if hands_intersect and intersect_confidence > 50.0:
        return "hands_intersect", intersect_confidence
    
    # Check for India gesture (Blue hand above shoulder + intersects face) - high priority
    india_detected, india_confidence = detect_india_gesture(results)
    if india_detected and india_confidence > 50.0:
        return "india", india_confidence
    
    # Check for Female gesture (Blue hand near nose) - high priority
    female_detected, female_confidence = detect_female_gesture(results)
    if female_detected and female_confidence > 50.0:
        return "female", female_confidence
    
    # Check for Male gesture (Blue hand near lips) - high priority
    male_detected, male_confidence = detect_male_gesture(results)
    if male_detected and male_confidence > 50.0:
        return "male", male_confidence
    
    # Check for Thank you gesture (Blue hand touching chin) - high priority
    thank_you_detected, thank_you_confidence = detect_thank_you_gesture(results)
    if thank_you_detected and thank_you_confidence > 50.0:
        return "thank_you", thank_you_confidence
    
    # Check for Sibling gesture (Middle finger touching same-side shoulder) - high priority
    sibling_detected, sibling_confidence = detect_sibling_gesture(results)
    if sibling_detected and sibling_confidence > 50.0:
        return "sibling", sibling_confidence
    
    # Check for Water gesture (Green hand inside mouth) - high priority
    water_detected, water_confidence = detect_water_gesture(results)
    if water_detected and water_confidence > 50.0:
        return "water", water_confidence
    
    # Note: Doctor gesture is handled separately in main loop with state tracking
    
    # Check for cheek gesture (has priority)
    cheek_detected, cheek_confidence = detect_cheek_gesture(results)
    if cheek_detected and cheek_confidence > 50.0:
        return "index_on_cheek", cheek_confidence
    
    # Check for both palms forward (has priority over single palm)
    both_palms, palms_confidence = detect_both_palms_forward(results)
    if both_palms and palms_confidence > 50.0:
        return "both_palms", palms_confidence
    
    if not left_hand_detected and not right_hand_detected:
        return None, 0.0
    
    left_thumb_up = False
    right_thumb_up = False
    left_thumb_down = False
    right_thumb_down = False
    left_palm_up = False
    right_palm_up = False
    
    # Process left hand
    if left_hand_detected:
        left_hand = results.left_hand_landmarks.landmark
        # Thumb up: thumb tip above IP and MCP
        left_thumb_up = left_hand[4].y < left_hand[3].y and left_hand[4].y < left_hand[2].y
        # Thumb down: thumb tip below IP and MCP
        left_thumb_down = left_hand[4].y > left_hand[3].y and left_hand[4].y > left_hand[2].y
        # Palm up: wrist below middle finger MCP
        left_palm_up = left_hand[0].y > left_hand[9].y
    
    # Process right hand
    if right_hand_detected:
        right_hand = results.right_hand_landmarks.landmark
        # Thumb up: thumb tip above IP and MCP
        right_thumb_up = right_hand[4].y < right_hand[3].y and right_hand[4].y < right_hand[2].y
        # Thumb down: thumb tip below IP and MCP
        right_thumb_down = right_hand[4].y > right_hand[3].y and right_hand[4].y > right_hand[2].y
        # Palm up: wrist below middle finger MCP
        right_palm_up = right_hand[0].y > right_hand[9].y
    
    # Gesture detection with priority
    # Check thumbs down first (has priority over thumbs up)
    if left_thumb_down or right_thumb_down:
        return "thumbs_down", 85.0  # "not agree"
    elif left_thumb_up and right_thumb_up:
        return "both_thumbs_up", 90.0  # "How are you?"
    elif left_thumb_up or right_thumb_up:
        return "one_thumb_up", 85.0  # "Okay"
    elif left_palm_up or right_palm_up:
        return "palm_up", 85.0  # "Hello"
    
    return None, 0.0

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_landmarks(image, results):
    """Draw MediaPipe landmarks on image"""
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Draw face landmarks (for cheek detection)
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1))
    
    # Draw left hand landmarks - BLUE color
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Blue (BGR)
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))  # Blue (BGR)
    
    # Draw right hand landmarks - GREEN color
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green (BGR)
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))  # Green (BGR)
    
    return image

def draw_prediction_info(image, buffer_active, buffer_remaining, current_word, confidence, gesture_detected, gesture_stable_time, gesture_confidence=0):
    """Draw prediction information on image - Beautiful and organized interface"""
    h, w = image.shape[:2]
    
    # Beautiful gradient-like overlay with better organization
    overlay = image.copy()
    # Top header bar
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 40), -1)
    # Main info area
    cv2.rectangle(overlay, (0, 85), (w, 220), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    # Beautiful title with colored words - "INDIAN SIGN LANGUAGE"
    # Colors: INDIAN = Orange, SIGN = White, LANGUAGE = Green
    words = ["INDIAN", "SIGN", "LANGUAGE"]
    colors = [
        (0, 165, 255),  # Orange (BGR format)
        (255, 255, 255),  # White
        (0, 255, 0)      # Green
    ]
    
    # Calculate positions for each word
    font_scale = 1.0
    font_thickness = 3
    x_start = 10
    y_pos = 35
    
    # Draw shadow first (all words together)
    full_title = "INDIAN SIGN LANGUAGE"
    cv2.putText(image, full_title, (12, 37), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4)
    
    # Draw each word with its color
    current_x = x_start
    for word, color in zip(words, colors):
        # Get text size to position next word
        (text_width, text_height), baseline = cv2.getTextSize(
            word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        # Draw the word
        cv2.putText(image, word, (current_x, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        # Move x position for next word (add space)
        current_x += text_width + 15  # 15 pixels spacing between words
    
    # Subtitle
    subtitle = "Real-Time Gesture Recognition System"
    cv2.putText(image, subtitle, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)  # Light gray
    
    # Status display (simplified, buffer/stability moved to bottom)
    if buffer_active:
        # Status text - organized layout
        cv2.putText(image, "Status:", (10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, "Preparing...", (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # Recognition mode - beautiful layout
        cv2.putText(image, "Status:", (10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, "ACTIVE", (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green
        
        # Show current gesture being detected - Beautiful organized layout
        if gesture_detected:
            gesture_names = {
                "namaste": "Prayer Hands (Namaste)",
                "hands_intersect": "Hands Intersect (Blue + Green)",
                "india": "Blue Thumb Above Eyebrows + Hand Above Shoulder",
                "female": "Blue Hand Touching Nose",
                "male": "Blue Hand Touching Lips",
                "thank_you": "Blue Hand Touching Chin",
                "sibling": "Middle Finger Touch Same Shoulder",
                "doctor": "Blue Hand Touching Shoulder Width",
                "water": "Green Hand Inside Mouth",
                "both_thumbs_up": "Both Thumbs Up",
                "one_thumb_up": "One Thumb Up",
                "thumbs_down": "Thumbs Down",
                "palm_up": "Palm Up",
                "index_on_cheek": "Index on Right Cheek",
                "both_palms": "Both Palms Forward",
                "head_nod": "Head Nod",
                "head_rotate": "Head Rotate"
            }
            gesture_name = gesture_names.get(gesture_detected, gesture_detected)
            # Namaste requires 0.4 seconds, Water requires 0.8 seconds, others use 0.3 seconds
            if gesture_detected == "namaste":
                required_stability = 0.4
            elif gesture_detected == "water":
                required_stability = 0.8
            else:
                required_stability = 0.3
            stability = min(gesture_stable_time / required_stability, 1.0)
            stability_pct = int(stability * 100)
            
            # Organized gesture info box
            info_y = 105
            cv2.putText(image, "Detecting:", (10, info_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(image, gesture_name, (10, info_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)  # Cyan
            
            # Confidence display
            if gesture_confidence > 0:
                conf_y = info_y + 45
                cv2.putText(image, "Confidence:", (10, conf_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                conf_text = f"{gesture_confidence:.1f}%"
                cv2.putText(image, conf_text, (10, conf_y + 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)  # Orange
            
    
    # Current prediction - Beautiful organized display
    if current_word and confidence > 0:
        # Prediction box with beautiful styling - moved down to avoid overlap
        pred_y = 150  # Increased from 105 to avoid overlap with gesture detection
        cv2.putText(image, "RECOGNIZED:", (10, pred_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Word with shadow effect
        word_y = pred_y + 25
        # Shadow
        cv2.putText(image, current_word, (12, word_y + 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        # Main text with confidence-based color
        if confidence >= 80.0:
            text_color = (0, 255, 0)  # Bright green
        elif confidence >= 60.0:
            text_color = (0, 255, 255)  # Cyan
        else:
            text_color = (255, 165, 0)  # Orange
        cv2.putText(image, current_word, (10, word_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        # Confidence display
        conf_y = word_y + 30
        cv2.putText(image, "Confidence:", (10, conf_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        conf_text = f"{confidence:.1f}%"
        cv2.putText(image, conf_text, (10, conf_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    
    # Buffer and Stability info at bottom (slightly increased size)
    bottom_y = h - 15
    if buffer_active:
        buffer_text = f"Buffer: {buffer_remaining:.1f}s"
        cv2.putText(image, buffer_text, (10, bottom_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)  # Increased from 0.3 to 0.45
    else:
        if gesture_detected:
            stability = min(gesture_stable_time / 0.3, 1.0)  # 0.3 seconds for faster stability
            stability_pct = int(stability * 100)
            stability_text = f"Stability: {stability_pct}%"
            cv2.putText(image, stability_text, (10, bottom_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)  # Increased from 0.3 to 0.45
    
    return image

def main():
    print("="*70)
    print("GESTURE-BASED SIGN LANGUAGE RECOGNITION")
    print("="*70)
    
    # Gesture to word mapping
    gesture_map = {
        "namaste": "Namaste",
        "hands_intersect": "Home",
        "india": "I am Indian",
        "female": "Female",
        "male": "Male",
        "thank_you": "Thank you",
        "sibling": "Sibling",
        "doctor": "Doctor",
        "water": "i want water",
        "both_thumbs_up": "are you okay ?",
        "one_thumb_up": "Okay",
        "thumbs_down": "not agree",
        "palm_up": "Hello",
        "index_on_cheek": "Blind",
        "both_palms": "Big",
        "head_nod": "Yes",
        "head_rotate": "No"
    }
    
    print("\nGesture Mappings:")
    for gesture, word in gesture_map.items():
        print(f"  {gesture}: {word}")
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam ready!")
    print("\n" + "="*70)
    print("Starting gesture recognition...")
    print("Instructions:")
    print("  - Wait 4 seconds (buffer time)")
    print("  - Then show gesture and hold steady for 0.3 seconds")
    print("  - Prayer Hands (Namaste) = 'Namaste'")
    print("  - Hands Intersect (Blue + Green) = 'Home'")
    print("  - Blue Thumb Above Eyebrows + Hand Above Shoulder = 'I am Indian'")
    print("  - Blue Hand Touching Nose = 'Female'")
    print("  - Blue Hand Touching Lips = 'Male'")
    print("  - Blue Hand Touching Chin = 'Thank you'")
    print("  - Middle Finger Touch Same Shoulder = 'Sibling'")
    print("  - Blue Hand Touching Shoulder Width = 'Doctor'")
    print("  - Green Hand Inside Mouth (hold long) = 'i want water'")
    print("  - Both Thumbs Up = 'are you okay ?'")
    print("  - One Thumb Up = 'Okay'")
    print("  - Thumbs Down = 'not agree'")
    print("  - Palm Up = 'Hello'")
    print("  - Index Finger on Right Cheek = 'Blind'")
    print("  - Both Palms Forward = 'Big'")
    print("  - Head Nod (Up/Down) = 'Yes'")
    print("  - Head Rotate (Left/Right) = 'No'")
    print("  - Press 'q' to quit")
    print("="*70 + "\n")
    
    # Initialize MediaPipe
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Buffer and gesture tracking
    BUFFER_TIME = 4.0  # 4 seconds buffer
    GESTURE_STABLE_DURATION = 0.3  # Gesture must be stable for 0.3 seconds (faster)
    
    buffer_start_time = None
    buffer_active = True
    gesture_detected = None
    gesture_stable_time = 0
    current_word = None
    current_confidence = 0
    head_history = deque(maxlen=10)  # Track head position history for gesture detection
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Process with MediaPipe
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Draw landmarks
            frame = draw_landmarks(frame, results)
            
            # BUFFER TIME LOGIC
            current_time = time.time()
            
            if buffer_active:
                if buffer_start_time is None:
                    buffer_start_time = current_time
                    print("[BUFFER] Starting 4-second buffer...")
                
                elapsed = current_time - buffer_start_time
                remaining = BUFFER_TIME - elapsed
                
                if remaining > 0:
                    # Still in buffer - show countdown
                    buffer_remaining = remaining
                else:
                    # Buffer complete - start gesture recognition
                    buffer_active = False
                    buffer_start_time = None
                    print("[BUFFER] Complete! Starting gesture recognition...")
                    buffer_remaining = 0
            
            # GESTURE RECOGNITION (only after buffer)
            if not buffer_active:
                # First check for head gestures (has priority)
                head_gesture, head_confidence = detect_head_gesture(results, head_history)
                
                if head_gesture and head_confidence > 50.0:
                    detected_gesture = head_gesture
                    gesture_confidence = head_confidence
                else:
                    # Check Doctor gesture (hardcore brute force - blue hand touches shoulder width)
                    doctor_detected, doctor_conf = detect_doctor_gesture(results)
                    
                    if doctor_detected and doctor_conf > 50.0:
                        detected_gesture = "doctor"
                        gesture_confidence = doctor_conf
                    else:
                        # Check other hand gestures
                        detected_gesture, gesture_confidence = detect_gesture(results)
                
                if detected_gesture and gesture_confidence > 40.0:
                    if gesture_detected == detected_gesture:
                        # Same gesture - check if stable
                        gesture_stable_time += 0.03  # ~30ms per frame
                        
                        # Namaste requires 0.4 seconds, Water requires 0.8 seconds (long time), others use 0.3 seconds
                        if detected_gesture == "namaste":
                            required_stability = 0.4
                        elif detected_gesture == "water":
                            required_stability = 0.8  # Longer duration for "for a long time"
                        else:
                            required_stability = GESTURE_STABLE_DURATION
                        
                        if gesture_stable_time >= required_stability:
                            # Gesture is stable - recognize it
                            if detected_gesture in gesture_map:
                                recognized_word = gesture_map[detected_gesture]
                                
                                # Only update if different word
                                if recognized_word != current_word:
                                    current_word = recognized_word
                                    current_confidence = min(gesture_confidence, 95.0)
                                    print(f"[GESTURE] Detected: {recognized_word} (from {detected_gesture})")
                                    print(f"   Confidence: {gesture_confidence:.1f}% | Stability: {gesture_stable_time:.2f}s")
                                    
                                    # Reset buffer for next gesture
                                    buffer_active = True
                                    buffer_start_time = current_time
                                    gesture_detected = None
                                    gesture_stable_time = 0
                                    print("[BUFFER] Resetting for next gesture...")
                    else:
                        # New gesture detected - reset stability timer
                        gesture_detected = detected_gesture
                        gesture_stable_time = 0
                else:
                    # No gesture detected or low confidence - reset
                    gesture_detected = None
                    gesture_stable_time = 0
            
            # Draw UI
            buffer_remaining_display = buffer_remaining if buffer_active else 0
            gesture_conf_display = gesture_confidence if 'gesture_confidence' in locals() else 0
            frame = draw_prediction_info(
                frame, buffer_active, buffer_remaining_display,
                current_word, current_confidence, gesture_detected, gesture_stable_time, gesture_conf_display
            )
            
            # Show frame
            cv2.imshow('Sign Language Recognition - Gesture Mode', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("Cleaned up resources")

if __name__ == '__main__':
    main()

