import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from collections import deque

# Import from inference_camera
from inference_camera import PoseDetector, reduce_keypoints, add_velocity

def main():
    print("Starting velocity checker...")
    print("This will show you the velocity magnitudes in real-time")
    print("Stand still to see if velocities are near zero")
    print("Walk in place to see velocity increase\n")
    
    # Initialize pose detector
    pose_detector = PoseDetector()
    cam = cv2.VideoCapture(0)
    
    pose_buffer = deque(maxlen=30)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose_detector.detect(frame_rgb)
        
        mean_confidence = keypoints[0, 0, :, 2].mean()
        
        if mean_confidence > 0.3:
            pose_buffer.append(keypoints[0, 0])
            
            if len(pose_buffer) == 30:
                pose_array = np.array(list(pose_buffer))
                
                # Reduce keypoints
                X = pose_array.copy()
                X = X[..., :-1]  # Remove confidence
                X = reduce_keypoints(X)  # 18→13
                
                # Add velocity
                Xv = add_velocity(X)  # (30, 13, 4) - last 2 channels are velocity
                
                # Get velocity magnitudes
                velocities = Xv[:, :, 2:]  # (30, 13, 2) - vx, vy
                vel_magnitudes = np.linalg.norm(velocities, axis=-1)  # (30, 13)
                
                mean_vel = vel_magnitudes.mean()
                max_vel = vel_magnitudes.max()
                
                # Show on frame
                cv2.putText(frame, f"Mean Velocity: {mean_vel:.4f}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Max Velocity: {max_vel:.4f}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add guidance
                if mean_vel < 0.01:
                    status = "Very still (likely standing)"
                    color = (0, 255, 0)
                elif mean_vel < 0.05:
                    status = "Small motion (swaying/breathing)"
                    color = (0, 165, 255)
                else:
                    status = "Movement detected (walking/action)"
                    color = (0, 0, 255)
                
                cv2.putText(frame, status, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Velocity Check', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
