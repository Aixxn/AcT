# Real-time Camera Inference for Action Recognition

Real-time action recognition from webcam using the Action Transformer (AcT) model with MoveNet pose detection.

## Prerequisites

1. **MoveNet TFLite model**: Download and place in `bin/movenet_256.tflite`
   - Download from: https://www.kaggle.com/models/google/movenet/frameworks/tfLite/variations/singlepose-lightning/versions/4
   - Or use the TensorFlow Hub version

2. **Trained AcT weights**: Already in `bin/openpose/micro/AcT_micro.h5`

3. **Python packages**: Ensure opencv-python is installed
   ```bash
   pip install opencv-python
   ```

## Usage

### Basic Usage (Default Camera)

```bash
python inference_camera.py
```

### With Pose Visualization

```bash
python inference_camera.py --show-pose
```

### Custom Camera

```bash
python inference_camera.py --camera 1
```

### Custom Model Weights

```bash
python inference_camera.py --weights bin/openpose/micro/AcT_micro.h5
```

### All Options

```bash
python inference_camera.py \
    --config utils/config.yaml \
    --weights bin/openpose/micro/AcT_micro.h5 \
    --camera 0 \
    --movenet bin/movenet_256.tflite \
    --show-pose \
    --window-size 30
```

## Command-Line Arguments

- `--config`: Config file path (default: `utils/config.yaml`)
- `--weights`: Model weights path (overrides config)
- `--camera`: Camera device index (default: `0`)
- `--movenet`: MoveNet TFLite model path (default: `bin/movenet_256.tflite`)
- `--show-pose`: Show pose skeleton overlay
- `--window-size`: Number of frames for action window (default: `30`)

## Controls

- **Press 'q'**: Quit the application

## How It Works

1. **Pose Detection**: MoveNet extracts 17 body keypoints from each frame
2. **Sliding Window**: Maintains a buffer of 30 frames (configurable)
3. **Preprocessing**: 
   - Reduces 17 keypoints to 13 (averaging head and feet)
   - Adds velocity features
   - Normalizes and centers poses
4. **Action Recognition**: AcT model predicts action from pose sequence
5. **Visualization**: Displays prediction with confidence score

## Output Display

- **Top**: Predicted action and confidence score
  - Green: Confidence > 50%
  - Orange: Confidence < 50%
- **Middle**: Pose skeleton (if `--show-pose` enabled)
- **Bottom Left**: Buffer status (frames collected / window size)
- **Bottom Right**: FPS counter

## Recognized Actions

1. standing
2. check-watch
3. cross-arms
4. scratch-head
5. sit-down
6. get-up
7. turn-around
8. walking
9. wave1
10. boxing
11. kicking
12. pointing
13. pick-up
14. bending
15. hands-clapping
16. wave2
17. jogging
18. jumping
19. pjump (power jump)
20. running

## Tips for Best Results

1. **Lighting**: Ensure good, even lighting for better pose detection
2. **Background**: Plain background helps pose detection accuracy
3. **Distance**: Stand 2-3 meters from camera for full body visibility
4. **Performance**: 
   - Use GPU (CUDA + cuDNN) for faster inference
   - Reduce window size for lower latency (but less accuracy)
   - Close other applications to improve FPS

## Troubleshooting

**MoveNet model not found**:
```bash
# Download movenet_256.tflite and place in bin/
mkdir -p bin
# Download from TensorFlow Hub or Kaggle
```

**Low FPS**:
- Install CUDA + cuDNN for GPU acceleration
- Reduce camera resolution
- Use a smaller model variant

**No pose detected**:
- Check lighting and camera angle
- Ensure full body is visible
- Adjust pose detection threshold in code

**Action predictions incorrect**:
- Ensure proper distance from camera
- Perform actions clearly and distinctly
- Wait for buffer to fill (30 frames)

## Advanced Configuration

Edit `inference_camera.py` to customize:

- `threshold` in `draw_keypoints()`: Keypoint confidence threshold
- `mean_confidence > 0.3`: Pose detection sensitivity
- Preprocessing functions: Adjust normalization parameters

## Performance Notes

- **CPU-only**: ~5-10 FPS (depends on hardware)
- **GPU (CUDA)**: ~20-30 FPS
- **Latency**: ~1-2 seconds (30-frame window)

For lower latency, reduce `--window-size` (but may affect accuracy).
