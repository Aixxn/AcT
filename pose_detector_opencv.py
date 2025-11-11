"""
TensorFlow Lite MoveNet pose detector (no MediaPipe dependency)
"""
import cv2
import numpy as np
from pathlib import Path


class TFLitePoseDetector:
    """Pose detection using TensorFlow Lite MoveNet model"""
    
    def __init__(self):
        """Initialize TFLite MoveNet pose detector"""
        try:
            import tensorflow as tf
            
            # Try to find or download MoveNet model
            model_path = Path("models/movenet")
            model_path.mkdir(parents=True, exist_ok=True)
            model_file = model_path / "movenet_thunder.tflite"
            
            if not model_file.exists():
                print("Downloading MoveNet model...")
                self._download_movenet(model_file)
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=str(model_file))
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # MoveNet outputs 17 keypoints
            self.n_keypoints = 17
            
            print("MoveNet model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TFLite detector: {e}")
    
    def _download_movenet(self, model_file):
        """Download MoveNet Thunder model"""
        import urllib.request
        
        url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
        
        try:
            print(f"Downloading to {model_file}...")
            urllib.request.urlretrieve(url, model_file)
            print("Download complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to download MoveNet model: {e}")
    
    def detect(self, frame):
        """
        Detect pose using MoveNet
        Args:
            frame: RGB image (H, W, 3)
        Returns:
            keypoints (1, 1, 18, 3) - OpenPose format
        """
        # Resize for MoveNet (expects 256x256)
        h, w = frame.shape[:2]
        input_image = cv2.resize(frame, (256, 256))
        
        # Check model input type and prepare accordingly
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.uint8:
            # Model expects uint8 (0-255)
            input_image = input_image.astype(np.uint8)
        else:
            # Model expects float32 (normalized)
            input_image = input_image.astype(np.float32)
        
        input_image = np.expand_dims(input_image, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # MoveNet output: [1, 1, 17, 3] (y, x, score)
        movenet_kpts = keypoints_with_scores[0, 0]
        
        # Convert MoveNet 17 keypoints to OpenPose 18 format
        openpose_kpts = self._movenet_to_openpose(movenet_kpts)
        
        return openpose_kpts.reshape(1, 1, 18, 3)
    
    def _movenet_to_openpose(self, movenet_kpts):
        """
        Convert MoveNet 17 keypoints to OpenPose 18 format
        MoveNet order: nose, left_eye, right_eye, left_ear, right_ear,
                       left_shoulder, right_shoulder, left_elbow, right_elbow,
                       left_wrist, right_wrist, left_hip, right_hip,
                       left_knee, right_knee, left_ankle, right_ankle
        """
        openpose = np.zeros((18, 3))
        
        # MoveNet outputs (y, x, score) - need to swap to (x, y, score)
        movenet_xy = movenet_kpts[:, [1, 0, 2]]  # Swap y,x to x,y
        
        # Map MoveNet indices to OpenPose indices
        # OpenPose: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist,
        #           MidHip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar
        
        openpose[0] = movenet_xy[0]   # Nose
        openpose[2] = movenet_xy[6]   # RShoulder
        openpose[3] = movenet_xy[8]   # RElbow
        openpose[4] = movenet_xy[10]  # RWrist
        openpose[5] = movenet_xy[5]   # LShoulder
        openpose[6] = movenet_xy[7]   # LElbow
        openpose[7] = movenet_xy[9]   # LWrist
        openpose[9] = movenet_xy[12]  # RHip
        openpose[10] = movenet_xy[14] # RKnee
        openpose[11] = movenet_xy[16] # RAnkle
        openpose[12] = movenet_xy[11] # LHip
        openpose[13] = movenet_xy[13] # LKnee
        openpose[14] = movenet_xy[15] # LAnkle
        openpose[15] = movenet_xy[2]  # REye
        openpose[16] = movenet_xy[1]  # LEye
        openpose[17] = movenet_xy[4]  # REar
        # LEar at index 18 doesn't exist in OpenPose 18-point format
        
        # Compute Neck (average of shoulders)
        if openpose[2, 2] > 0.1 and openpose[5, 2] > 0.1:
            openpose[1] = (openpose[2] + openpose[5]) / 2
        
        # Compute MidHip (average of hips)
        if openpose[9, 2] > 0.1 and openpose[12, 2] > 0.1:
            openpose[8] = (openpose[9] + openpose[12]) / 2
        
        return openpose
