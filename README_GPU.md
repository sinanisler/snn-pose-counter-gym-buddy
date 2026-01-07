# 🚀 GPU-Accelerated YOLO Exercise Counter

Real-time pose detection using YOLOv8n-pose with **NVIDIA CUDA acceleration** for maximum performance.

## 🎯 Performance

- **CPU-only (browser)**: 5-10 FPS
- **GPU-accelerated (Python backend)**: 30-60+ FPS ⚡

## 📋 Prerequisites

### 1. CUDA and cuDNN
You need CUDA 12.x and cuDNN 9.x installed:

- **CUDA 12.x**: Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- **cuDNN 9.x**: Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

Or use the simplified installation:
```bash
pip install onnxruntime-gpu[cuda,cudnn]
```

### 2. Python 3.8+
Make sure Python is installed on your system.

### 3. YOLO Model
You already have the `yolov8n-pose.onnx` model in this directory.

## 🔧 Installation

### Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Verify CUDA Installation

Run this to check if CUDA is available:

```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
```

You should see `['CUDAExecutionProvider', 'CPUExecutionProvider']` if CUDA is properly installed.

## 🚀 Usage

### Step 1: Start the Backend Server

```bash
python backend.py
```

You should see:
```
============================================================
GPU-Accelerated YOLO Pose Estimation Backend
============================================================
Initializing YOLO model with CUDA...
Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Using providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
✓ Successfully initialized with CUDA GPU acceleration!

Starting Flask server on http://localhost:5000
============================================================
```

### Step 2: Open the Frontend

Open `index_gpu.html` in your web browser (Chrome or Edge recommended).

### Step 3: Start Exercising!

1. Click "Start Webcam" (allow camera access)
2. Select your exercise type (Squats, Bicep Curls, or Push-ups)
3. Make sure your whole body is visible
4. Start exercising and watch the counter!

## 📊 Features

- ✅ **GPU Acceleration**: Runs on NVIDIA CUDA for 5-10x faster inference
- ✅ **Real-time Performance**: 30-60+ FPS on RTX 3060 (12GB)
- ✅ **Multiple Exercises**: Squats, Bicep Curls, Push-ups
- ✅ **Performance Monitoring**: Live FPS and inference time display
- ✅ **GPU Stats**: Optional GPU memory and utilization monitoring

## 🔍 Troubleshooting

### Backend doesn't start

**Issue**: `CUDA provider not available`

**Solution**: Make sure CUDA and cuDNN are installed correctly. You can also run:
```bash
pip install onnxruntime-gpu[cuda,cudnn]
```

### Low FPS despite using GPU

**Issue**: GPU not being utilized

**Solution**:
1. Check if CUDA provider is being used in the backend logs
2. Make sure no other applications are using the GPU heavily
3. Try closing other browser tabs/applications

### Cannot connect to backend

**Issue**: `Cannot connect to backend on port 5000`

**Solution**:
1. Make sure `backend.py` is running
2. Check if another application is using port 5000
3. Try changing the port in both `backend.py` and `index_gpu.html`

## 🛠️ Architecture

```
┌─────────────┐         HTTP          ┌──────────────┐
│   Browser   │ ◄──────────────────► │    Flask     │
│  (Frontend) │   Send frame data     │   Backend    │
│             │   Receive keypoints   │              │
└─────────────┘                       └──────┬───────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ ONNX Runtime │
                                      │  + CUDA EP   │
                                      └──────┬───────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │  NVIDIA GPU  │
                                      │  RTX 3060    │
                                      │  (12GB VRAM) │
                                      └──────────────┘
```

## 📝 API Endpoints

### `GET /health`
Check if backend and model are ready.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "cuda_available": true,
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}
```

### `POST /process_frame`
Process a single video frame.

**Request:**
```json
{
  "frame": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "detection": {
    "keypoints": [...],
    "score": 0.95
  },
  "timing": {
    "preprocess_ms": 2.5,
    "inference_ms": 8.3,
    "postprocess_ms": 1.2,
    "total_ms": 12.0,
    "fps": 83.3
  }
}
```

### `GET /stats` (Optional)
Get GPU utilization statistics.

**Response:**
```json
{
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "memory_used_mb": 512.5,
  "memory_total_mb": 12288.0,
  "memory_percent": 4.17,
  "gpu_utilization": 45,
  "temperature": 62
}
```

## 🎯 Exercise Detection Logic

### Squats
- **Down**: Knee angle < 100°
- **Up**: Knee angle > 160°
- **Count**: One complete down-up cycle

### Bicep Curls
- **Down**: Elbow angle > 160° (extended)
- **Curled**: Elbow angle < 50°
- **Count**: One complete down-up cycle

### Push-ups
- **Down**: Elbow angle < 90°
- **Up**: Elbow angle > 160°
- **Count**: One complete down-up cycle

## 📦 Files

- `backend.py` - Flask server with CUDA-accelerated YOLO inference
- `index_gpu.html` - GPU-accelerated frontend
- `index.html` - Original CPU-only version (for comparison)
- `requirements.txt` - Python dependencies
- `yolov8n-pose.onnx` - YOLO pose estimation model

## 🔗 Resources

- [ONNX Runtime CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## 📄 License

MIT License - feel free to use this for your projects!

---

**Enjoy your GPU-accelerated workout tracking! 💪🚀**
