"""
GPU-Accelerated YOLO Pose Estimation Backend
Uses ONNX Runtime with CUDA for high-performance inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
import base64
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables
session = None
model_input_size = (640, 640)

def initialize_model():
    """Initialize ONNX Runtime session with CUDA execution provider"""
    global session

    print("Initializing YOLO model with CUDA...")
    print(f"Available providers: {ort.get_available_providers()}")

    try:
        # Try CUDA first (GPU acceleration)
        session = ort.InferenceSession(
            'yolov8n-pose.onnx',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Check which provider is actually being used
        providers_used = session.get_providers()
        print(f"Using providers: {providers_used}")

        if 'CUDAExecutionProvider' in providers_used:
            print("✓ Successfully initialized with CUDA GPU acceleration!")
        else:
            print("⚠ Warning: CUDA not available, falling back to CPU")

        # Print model info
        print(f"Model inputs: {session.get_inputs()[0].name}")
        print(f"Model input shape: {session.get_inputs()[0].shape}")
        print(f"Model outputs: {session.get_outputs()[0].name}")

        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def preprocess_frame(frame):
    """Preprocess frame for YOLO model input"""
    # Resize to model input size
    img = cv2.resize(frame, model_input_size)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and convert to float32
    img = img.astype(np.float32) / 255.0

    # Transpose from HWC to CHW format
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension [1, 3, 640, 640]
    img = np.expand_dims(img, axis=0)

    return img

def postprocess_output(output, original_width, original_height, conf_threshold=0.5):
    """Process YOLO output to extract keypoints"""
    # Output shape: [1, 56, 8400]
    # Format: [x, y, w, h, confidence, 17 keypoints * 3 (x, y, conf)]

    output = output[0]  # Remove batch dimension
    num_detections = output.shape[1]

    best_detection = None
    best_score = 0

    for i in range(num_detections):
        # Get confidence score (index 4)
        score = output[4, i]

        if score > conf_threshold and score > best_score:
            best_score = score

            # Get bounding box
            x = float(output[0, i])
            y = float(output[1, i])
            w = float(output[2, i])
            h = float(output[3, i])

            # Get keypoints (17 keypoints * 3 values = 51 values)
            keypoints = []
            for k in range(17):
                kp_x = float(output[5 + k * 3, i]) * original_width / model_input_size[0]
                kp_y = float(output[5 + k * 3 + 1, i]) * original_height / model_input_size[1]
                kp_conf = float(output[5 + k * 3 + 2, i])

                keypoints.append({
                    'x': kp_x,
                    'y': kp_y,
                    'confidence': kp_conf
                })

            best_detection = {
                'box': {'x': x, 'y': y, 'w': w, 'h': h},
                'score': float(score),
                'keypoints': keypoints
            }

    return best_detection

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': session is not None,
        'cuda_available': 'CUDAExecutionProvider' in ort.get_available_providers(),
        'providers': session.get_providers() if session else []
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame and return pose keypoints"""
    try:
        start_time = time.time()

        # Get frame from request
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode base64 frame
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400

        original_height, original_width = frame.shape[:2]

        # Preprocess
        preprocess_start = time.time()
        input_tensor = preprocess_frame(frame)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Run inference on GPU
        inference_start = time.time()
        outputs = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: input_tensor}
        )
        inference_time = (time.time() - inference_start) * 1000

        # Postprocess
        postprocess_start = time.time()
        detection = postprocess_output(outputs[0], original_width, original_height)
        postprocess_time = (time.time() - postprocess_start) * 1000

        total_time = (time.time() - start_time) * 1000

        return jsonify({
            'success': True,
            'detection': detection,
            'timing': {
                'preprocess_ms': float(round(preprocess_time, 2)),
                'inference_ms': float(round(inference_time, 2)),
                'postprocess_ms': float(round(postprocess_time, 2)),
                'total_ms': float(round(total_time, 2)),
                'fps': float(round(1000 / total_time, 1))
            }
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get GPU and model statistics"""
    try:
        import pynvml
        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        return jsonify({
            'gpu_name': gpu_name,
            'memory_used_mb': memory_info.used / (1024 ** 2),
            'memory_total_mb': memory_info.total / (1024 ** 2),
            'memory_percent': (memory_info.used / memory_info.total) * 100,
            'gpu_utilization': utilization.gpu,
            'temperature': temperature
        })
    except Exception as e:
        return jsonify({'error': f'Could not get GPU stats: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("GPU-Accelerated YOLO Pose Estimation Backend")
    print("=" * 60)

    # Initialize model
    if not initialize_model():
        print("Failed to initialize model. Exiting...")
        exit(1)

    print("\nStarting Flask server on http://localhost:5000")
    print("Frontend should connect to this backend for GPU-accelerated inference")
    print("=" * 60)

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
