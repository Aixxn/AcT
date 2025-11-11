import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import collections
import time
import argparse
from pathlib import Path


import cv2
import numpy as np
import tensorflow as tf

from utils.tools import read_yaml
from utils.trainer import Trainer

# Action class labels
CLASSES = [
    "standing", "check-watch", "cross-arms", "scratch-head", "sit-down",
    "get-up", "turn-around", "walking", "wave1", "boxing",
    "kicking", "pointing", "pick-up", "bending", "hands-clapping",
    "wave2", "jogging", "jumping", "pjump", "running"
]

# OpenPose keypoint mapping (25 keypoints total, we use 18 body keypoints)
OPENPOSE_KEYPOINTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18
}

# Skeleton edges for OpenPose visualization
KEYPOINT_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18)
]

# Preprocessing constants for OpenPose (from reference code)
# OpenPose uses different indexing than MoveNet
C1, C2 = 2, 5  # Right/Left shoulder as center points
M1, M2 = 9, 12  # Right/Left hip as module points
H = [0, 15, 16, 17]  # Head keypoints (Nose, REye, LEye, REar) - index 18 doesn't exist
RF, LF = [11], [14]  # Right/Left ankle


class PoseDetector:
    """Detects poses using TFLite MoveNet (no MediaPipe dependency)"""
    def __init__(self):
        # Try TFLite MoveNet first (best option)
        try:
            from pose_detector_opencv import TFLitePoseDetector
            self.detector = TFLitePoseDetector()
            print("Using TensorFlow Lite MoveNet pose detector")
        except Exception as e:
            print(f"TFLite MoveNet not available ({e})")
            print("Please ensure you have internet connection for model download")
            raise
        
    def detect(self, frame):
        """
        Run pose detection on frame
        Args:
            frame: RGB image (H, W, 3)
        Returns:
            keypoints with shape (1, 1, 18, 3) - [x, y, confidence] in OpenPose format
        """
        return self.detector.detect(frame)


class ActionRecognizer:
    """Recognizes actions using the AcT model"""
    def __init__(self, config_path, weights_path=None):
        self.config = read_yaml(config_path)
        
        if weights_path:
            self.config['WEIGHTS'] = weights_path
        
        # Build and load model
        print(f"Loading AcT model from: {self.config['WEIGHTS']}")
        trainer = Trainer(self.config, logger=None)
        
        # Create dummy data to initialize trainer
        trainer.train_len = 1000
        trainer.test_len = 100
        trainer.get_model()
        
        self.model = trainer.model
        self.model.load_weights(self.config['WEIGHTS'])
        
        self.n_frames = self.config[self.config['DATASET']]['FRAMES']
        self.keypoints = self.config[self.config['DATASET']]['KEYPOINTS']
        self.channels = self.config['CHANNELS']
        
        print(f"Model loaded. Window size: {self.n_frames} frames")
    
    def predict(self, pose_sequence):
        """
        Predict action from pose sequence
        Args:
            pose_sequence: preprocessed pose data (n_frames, features)
        Returns:
            (class_name, confidence)
        """
        if pose_sequence.shape[0] != self.n_frames:
            return None, 0.0
        
        # Predict
        logits = self.model.predict(pose_sequence[np.newaxis, ...], verbose=0)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        return CLASSES[pred_idx], confidence


def preprocess_pose(pose_sequence):
    """
    Preprocess raw OpenPose poses to AcT input format
    Args:
        pose_sequence: (T, 18, 3) array from OpenPose/MediaPipe
    Returns:
        (T, keypoints*channels) flattened and preprocessed
    """
    X = pose_sequence.copy()
    
    # Remove confidence, keep x, y
    X = X[..., :-1]  # (T, 18, 2)
    
    # Reduce keypoints from 18 to 13 (average head and feet)
    X = reduce_keypoints(X)  # (T, 13, 2)
    
    # Add velocity
    X = add_velocity(X)  # (T, 13, 4)
    
    # Scale and center
    X = scale_and_center(X)  # (T, 13, 4)
    
    # Flatten
    X = X.reshape(X.shape[0], -1)  # (T, 52)
    
    return X


def reduce_keypoints(X):
    """
    Reduce 18 OpenPose keypoints to 13 by averaging head and feet groups,
    then removing Neck and MidHip (which are computed from other keypoints)
    """
    # First, average head keypoints into Nose (index 0)
    to_prune = []
    for group in [H, RF, LF]:
        if len(group) > 1:
            to_prune.append(group[1:])
    to_prune = [item for sublist in to_prune for item in sublist]
    
    # Average groups
    X[:, H[0], :] = np.true_divide(X[:, H].sum(1), (X[:, H] != 0).sum(1) + 1e-9)
    X[:, RF[0], :] = np.true_divide(X[:, RF].sum(1), (X[:, RF] != 0).sum(1) + 1e-9)
    X[:, LF[0], :] = np.true_divide(X[:, LF].sum(1), (X[:, LF] != 0).sum(1) + 1e-9)
    
    # Remove averaged keypoints
    Xr = np.delete(X, to_prune, 1)  # 18 → 15 keypoints
    
    # Remove Neck (index 1) and MidHip (index 8) as they're computed from other points
    # After first deletion, indices shift, so we need to be careful
    # Original indices: 0(Nose), 1(Neck), 2(RShoulder), ..., 8(MidHip), ...
    # After removing eyes/ears (indices 15,16,17), we have 15 keypoints
    # Neck is still at index 1, MidHip is still at index 8
    Xr = np.delete(Xr, [1, 8], 1)  # 15 → 13 keypoints
    
    return Xr


def add_velocity(X):
    """Add velocity features"""
    T, K, C = X.shape
    v1, v2 = np.zeros((T + 1, K, C)), np.zeros((T + 1, K, C))
    v1[1:] = X
    v2[:T] = X
    vel = (v2 - v1)[:-1]
    Xv = np.concatenate((X, vel), axis=-1)
    return Xv


def scale_and_center(X):
    """Scale and center poses"""
    pose_list = []
    for pose in X:
        zero_point = (pose[C1, :2] + pose[C2, :2]) / 2
        module_keypoint = (pose[M1, :2] + pose[M2, :2]) / 2
        scale_mag = np.linalg.norm(zero_point - module_keypoint)
        if scale_mag < 1:
            scale_mag = 1
        pose[:, :2] = (pose[:, :2] - zero_point) / scale_mag
        pose_list.append(pose)
    Xn = np.stack(pose_list)
    return Xn


def draw_keypoints(frame, keypoints, threshold=0.3):
    """Draw keypoints and skeleton on frame (OpenPose format)"""
    h, w = frame.shape[:2]
    
    # Extract keypoint positions
    kpts = keypoints[0, 0, :, :]  # (18, 3)
    
    # Draw skeleton edges
    for edge in KEYPOINT_EDGES:
        if edge[0] < len(kpts) and edge[1] < len(kpts):
            if kpts[edge[0], 2] > threshold and kpts[edge[1], 2] > threshold:
                x1, y1 = int(kpts[edge[0], 0] * w), int(kpts[edge[0], 1] * h)
                x2, y2 = int(kpts[edge[1], 0] * w), int(kpts[edge[1], 1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints
    for i in range(len(kpts)):
        if kpts[i, 2] > threshold:
            x, y = int(kpts[i, 0] * w), int(kpts[i, 1] * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Real-time Action Recognition')
    parser.add_argument('--config', default='utils/config.yaml', help='Config file')
    parser.add_argument('--weights', default=None, help='Model weights (overrides config)')
    parser.add_argument('--camera', default='0', help='Camera index or URL (e.g., http://192.168.1.4:8080/video)')
    parser.add_argument('--show-pose', action='store_true', help='Show pose skeleton')
    parser.add_argument('--window-size', type=int, default=30, help='Frame window size')
    
    args = parser.parse_args()
    
    # Initialize pose detector
    print("Initializing pose detector...")
    try:
        pose_detector = PoseDetector()
    except Exception as e:
        print(f"Error: {e}")
        print("Pose detection initialization failed")
        return
    
    # Initialize action recognizer
    print("Initializing action recognizer...")
    action_recognizer = ActionRecognizer(args.config, args.weights)
    
    # Open camera (support both index and URL)
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)
    print(f"Opening camera: {camera_source}")
    
    cam = cv2.VideoCapture(camera_source)
    if not cam.isOpened():
        print(f"Error: Could not open camera {camera_source}")
        print("\nTroubleshooting:")
        print("  - For laptop webcam: --camera 0")
        print("  - For USB/external camera: --camera 1, 2, etc.")
        print("  - For IP Webcam (Android): --camera http://IP:8080/video")
        print("  - For DroidCam: Install DroidCam client, then use --camera 1 or 2")
        return
    
    print(f"Camera opened. Press 'q' to quit.")
    
    # Sliding window buffer
    pose_buffer = collections.deque(maxlen=args.window_size)
    
    # Prediction smoothing buffer (temporal filtering)
    prediction_history = collections.deque(maxlen=10)  # Last 10 predictions
    
    # Real-time velocity tracking (frame-to-frame)
    prev_pose = None
    current_velocity = 0.0
    velocity_history = collections.deque(maxlen=5)  # Smooth velocity over 5 frames
    
    # FPS counter
    fps_window = collections.deque(maxlen=30)
    prev_time = time.time()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Convert to RGB for pose detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        keypoints = pose_detector.detect(frame_rgb)
        
        # Check if pose detected with confidence
        mean_confidence = keypoints[0, 0, :, 2].mean()
        
        if mean_confidence > 0.3:
            # Add to buffer
            current_pose = keypoints[0, 0]  # (18, 3)
            pose_buffer.append(current_pose)
            
            # Calculate REAL-TIME velocity (frame-to-frame)
            if prev_pose is not None:
                # Compute velocity between current and previous frame
                curr_xy = current_pose[:, :2]  # (18, 2)
                prev_xy = prev_pose[:, :2]
                
                # Calculate displacement
                displacement = curr_xy - prev_xy  # (18, 2)
                
                # Compute velocity magnitude for each keypoint
                vel_per_keypoint = np.linalg.norm(displacement, axis=-1)  # (18,)
                
                # Average velocity across all keypoints
                frame_velocity = vel_per_keypoint.mean()
                
                # Add to velocity history for smoothing
                velocity_history.append(frame_velocity)
                
                # Smoothed velocity (average over last 5 frames)
                current_velocity = np.mean(velocity_history) if len(velocity_history) > 0 else 0.0
            
            # Update previous pose
            prev_pose = current_pose.copy()
            
            # Predict action when buffer is full
            if len(pose_buffer) == args.window_size:
                pose_array = np.array(list(pose_buffer))  # (T, 18, 3)
                
                # Preprocess
                preprocessed = preprocess_pose(pose_array)  # (T, 52)
                
                # Predict
                action, confidence = action_recognizer.predict(preprocessed)
                
                # Override walking prediction if velocity is too low (likely just standing still with jitter)
                # Use REAL-TIME velocity instead of buffer velocity for instant response
                # Threshold: standing still has velocity < 0.015, walking > 0.02
                if action == "walking" and current_velocity < 0.015:
                    # Replace with "standing" if it's in top predictions
                    logits = action_recognizer.model.predict(preprocessed[np.newaxis, ...], verbose=0)
                    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
                    
                    # Get standing class index
                    standing_idx = CLASSES.index("standing")
                    standing_conf = probs[standing_idx]
                    
                    # If standing is reasonably confident, use it instead
                    if standing_conf > 0.15:  # At least 15% confidence for standing
                        action = "standing"
                        confidence = standing_conf

                prediction_history.append((action, confidence, current_velocity))
                
                # Smooth predictions: use most common action in recent history
                # with high confidence threshold
                if len(prediction_history) >= 3:
                    recent_actions = [p[0] for p in prediction_history]
                    recent_confidences = [p[1] for p in prediction_history]
                    recent_velocities = [p[2] for p in prediction_history]
                    
                    # Get most common action
                    from collections import Counter
                    action_counts = Counter(recent_actions)
                    smoothed_action, count = action_counts.most_common(1)[0]
                    
                    # Only display if it appears in majority and has reasonable confidence
                    avg_confidence = np.mean([c for a, c, v in prediction_history if a == smoothed_action])
                    avg_velocity = np.mean(recent_velocities)
                    
                    # Higher threshold for walking to avoid false positives
                    # Also check REAL-TIME velocity - walking should have higher velocity
                    if smoothed_action == "walking":
                        if avg_confidence > 0.5 and current_velocity > 0.015 and count >= 2:
                            action = smoothed_action
                            confidence = avg_confidence
                        else:
                            # Not enough evidence for walking, default to standing
                            action = "standing"
                            confidence = avg_confidence
                    else:
                        # Other actions use normal threshold
                        min_confidence = 0.4
                        if avg_confidence > min_confidence and count >= 2:
                            action = smoothed_action
                            confidence = avg_confidence
                        else:
                            action = None
                
                # Display prediction
                if action:
                    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
                    text = f"{action}: {confidence:.2%}"
                    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                               1.2, color, 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No pose detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Reset velocity when no pose detected
            current_velocity = 0.0
            prev_pose = None
            velocity_history.clear()
        
        # Draw pose if requested
        if args.show_pose:
            frame = draw_keypoints(frame, keypoints)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        fps_window.append(fps)
        avg_fps = np.mean(fps_window)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show buffer status
        buffer_text = f"Buffer: {len(pose_buffer)}/{args.window_size}"
        cv2.putText(frame, buffer_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show REAL-TIME velocity (only if pose detected)
        if mean_confidence > 0.3:
            # Color coding: Green (still), Orange (small movement), Red (movement)
            # Velocity thresholds: < 0.01 = still, 0.01-0.02 = small, > 0.02 = moving
            if current_velocity < 0.01:
                vel_color = (0, 255, 0)  # Green - standing still
                vel_status = "Still"
            elif current_velocity < 0.02:
                vel_color = (0, 165, 255)  # Orange - small movement
                vel_status = "Small Motion"
            else:
                vel_color = (0, 0, 255)  # Red - active movement
                vel_status = "Moving"
            
            cv2.putText(frame, f"Velocity: {current_velocity:.4f} ({vel_status})", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2)
        
        # Display
        cv2.imshow('Action Recognition', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print("Camera released")


if __name__ == '__main__':
    main()
