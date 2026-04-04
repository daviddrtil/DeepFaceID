# DeepFaceID

Detection of Facial Deepfakes Using Interactive Liveness Tests

A real-time liveness detection system for remote identity verification that combines interactive tests (head rotation, facial occlusion, blinking, expression changes) with passive analysis (artifacts, textures, light reflections, micro-movements) to detect deepfake attacks, particularly face-swapping.

---

## Features

- **Interactive Liveness Tests**: Head rotation, facial occlusion, blinking, expression changes
- **Passive Detection**: Artifact analysis, texture inconsistencies, light reflection patterns, micro-movements
- **Real-time Processing**: GPU-accelerated inference for live camera streams
- **Web Interface**: Browser-based verification for remote identity checks
- **Evaluation Tools**: Metrics computation and static video analysis

---

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 5060 or equivalent (CUDA-capable)
- **CPU**: AMD Ryzen 7 250 or equivalent

### Software
- Python 3.11
- CUDA 12.8 (for GPU acceleration)
- FFmpeg
- Miniconda (recommended)

## Installation

1. **Install FFmpeg**
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add to system PATH

2. **Install Miniconda**
   - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)
   - Add to system PATH

3. **Create Python environment**
   ```bash
   conda create --name deepfaceid python=3.11
   conda activate deepfaceid
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify CUDA installation**
   ```bash
   python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
   ```

6. **Import pre-trained weights**
   - Download pre-trained weights from project release and place files into `/src/passive/weights/`

---

## Usage

### Live Verification Mode (Default)

Start the real-time verification system with web interface:

```bash
python src/run.py
```

The system runs in live mode with debug overlays enabled by default. Access the web interface at `http://<your-ip>:27027`

**Custom Configuration:**
```bash
# Specify network interface
python src/run.py --web-host=0.0.0.0 --web-port=27027

# Disable debug overlays
python src/run.py --no-debug

# Control visualization
python src/run.py --draw all          # Draw all landmarks
python src/run.py --draw face         # Draw face landmarks only
python src/run.py --draw hand         # Draw hand landmarks only
```

### Static Video Analysis

Process pre-recorded videos for evaluation and testing:

```bash
# Disable live mode for static analysis
python src/run.py --no-live --input-video path/to/video.mp4

# Analysis with frame sampling
python src/run.py --no-live --input-video video.mp4 --frame-sampling 30

# Quick test with frame limit
python src/run.py --no-live --input-video video.mp4 --max-frames 150

# Disable output saving
python src/run.py --no-live --input-video video.mp4 --no-saving-output
```

**Common Options:**
- `--live` / `--no-live`: Toggle live webcam mode (default: enabled)
- `--debug` / `--no-debug`: Toggle debug overlays and verbose logging (default: enabled)
- `--draw [all|face|hand]`: Render detection landmarks
- `--frame-sampling N`: Save every N-th processed frame as JPEG (default: 30)
- `--max-frames N`: Limit processing to N frames for quick tests
- `--no-saving-output`: Disable output video and frame saving

---

## License

MIT License - see [LICENSE](LICENSE) file for details

## Author

David Drtil
