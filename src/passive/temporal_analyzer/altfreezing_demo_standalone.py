import time
import math
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage import transform as trans

from training.detectors.altfreezing_detector import AltFreezingDetector

class VideoProcessor:
    def __init__(self, aligner, detector, smoothing_window=10):
        self.aligner = aligner
        self.detector = detector
        self.smoothing_window = smoothing_window

    def process(self, video_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_txt = output_dir / "scores.txt"
        output_video_path = output_dir / "output.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error. Could not open video file {video_path.name}")
            return

        print(f"Processing video {video_path.name}...")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        with open(output_txt, 'w') as file:
            frame_count = 0
            total_score = 0.0
            valid_frames = 0
            recent_scores = []

            clip_size = 8       # AltFreezing uses an 8-frame clip
            face_buffer = []
            frame_buffer_raw = []
            
            start_time = time.time()
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                    
                try:
                    face_crop = self.aligner.extract_and_align(frame)
                    face_buffer.append(face_crop)
                    frame_buffer_raw.append(frame)
                    
                    if len(face_buffer) == clip_size:
                        score = self.detector.predict(face_buffer)

                        recent_scores.append(score)
                        if len(recent_scores) > self.smoothing_window:
                            recent_scores.pop(0)

                        smoothed_score = sum(recent_scores) / len(recent_scores)

                        file.write(f"Frame {frame_count:3d} Raw={score:0.4f}, Smoothed={smoothed_score:0.4f}\n")

                        total_score += score
                        valid_frames += 1

                        face_filename = output_dir / f"face_{frame_count:04d}.jpg"
                        face_crop.save(face_filename)

                        current_frame = frame_buffer_raw[-1].copy()

                        text = f"Fake Score {smoothed_score*100:2.0f}%"
                        text_x = max(10, w - 350)
                        text_y = 50

                        cv2.putText(current_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(current_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        out_writer.write(current_frame)

                        if frame_count % 5 == 0:
                            print(f"Processed frame {frame_count:3d} Raw {score*100:2.0f}% Smoothed {smoothed_score*100:2.0f}%")

                        # remove the oldest frame to slide the window forward
                        face_buffer.pop(0)
                        frame_buffer_raw.pop(0)

                except Exception as e:
                    print(f"Failed on frame {frame_count}. Details are {e}")
                    file.write(f"Frame {frame_count} ERROR\n")
                    out_writer.write(frame)
                    # Clear buffers to avoid desync after an error
                    face_buffer.clear()
                    frame_buffer_raw.clear()

                frame_count += 1

            end_time = time.time()
            elapsed_seconds = end_time - start_time

            if valid_frames > 0:
                avg_score = total_score / valid_frames
                file.write(f"\nAverage Score {avg_score*100:.0f}%\n")
                print(f"\nAverage Fake Score for {video_path.name} {avg_score*100:.0f}%")
            else:
                file.write("\nAverage Score N/A (No faces detected)\n")
                print("\nCould not calculate average. No faces were detected.")

            if elapsed_seconds > 0:
                processing_fps = frame_count / elapsed_seconds
                print(f"Total processing time {elapsed_seconds:.2f} seconds")
                print(f"Processing speed {processing_fps:.2f} FPS")

        cap.release()
        out_writer.release()
        print(f"Finished processing {frame_count} frames.")
        print(f"Images saved to {output_dir.name} and video saved to {output_video_path.name}")


class FaceAligner:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract_and_align(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return Image.fromarray(img_rgb)

        landmarks = detection_result.face_landmarks[0]
        leye = [landmarks[468].x * w, landmarks[468].y * h]
        reye = [landmarks[473].x * w, landmarks[473].y * h]
        nose = [landmarks[1].x * w, landmarks[1].y * h]
        lmouth = [landmarks[61].x * w, landmarks[61].y * h]
        rmouth = [landmarks[291].x * w, landmarks[291].y * h]

        if leye[0] > reye[0]:
            leye, reye = reye, leye
        if lmouth[0] > rmouth[0]:
            lmouth, rmouth = rmouth, lmouth

        src_pts = np.array([leye, reye, nose, lmouth, rmouth], dtype=np.float32)

        outsize = [256, 256]
        scale = 1.3
        target_size = [112, 112]

        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        dst[:, 0] += 8.0
        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, dst)
        M = tform.params[0:2, :]
        aligned_face = cv2.warpAffine(img_rgb, M, (target_size[1], target_size[0]), flags=cv2.INTER_CUBIC)
        return Image.fromarray(aligned_face)


class AltFreezingVideoDetector:
    def __init__(self, config_path, weights_path, device):
        print("Loading AltFreezing model...")
        self.device = device

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model = AltFreezingDetector(config)

        weights = torch.load(weights_path, map_location="cpu")
        if 'net' in weights:
            weights = weights['net']
        elif 'state_dict' in weights:
            weights = weights['state_dict']

        self.model.load_state_dict(weights, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # TODO: AltFreezing uses 224x224 input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, face_images):
        tensors = []
        for img in face_images:
            tensors.append(self.transform(img))

        #(Channels, Time, Height, Width)
        video_tensor = torch.stack(tensors, dim=1)

        input_tensor = video_tensor.unsqueeze(0).to(self.device)
        dummy_label = torch.tensor([0]).to(self.device)

        data_dict = {
            'image': input_tensor,
            'label': dummy_label
        }

        with torch.no_grad():
            output = self.model(data_dict, inference=True)

        if 'prob' in output:
            probability = output['prob']
        elif 'cls' in output:
            logits = output['cls']
            probability = torch.softmax(logits, dim=1)[:, 1]
        else:
            raise KeyError(f"Unexpected output keys from model. Keys are {output.keys()}")
            
        if isinstance(probability, torch.Tensor):
            if probability.numel() > 1:
                return probability.mean().item()
            return probability.item()
        else:
            return probability[0]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    script_dir = Path(__file__).resolve().parent
    inputs_dir = script_dir / "inputs"

    weights_file = inputs_dir / "I3D_8x8_R50.pth"
    config_file = script_dir / "training/config/detector/altfreezing.yaml"
    landmarker_task = inputs_dir / "face_landmarker.task"

    video_name = "tomcruise.mp4"
    test_video = inputs_dir / video_name
    output_folder = script_dir / "outputs_altfreezing" / f"output_{video_name.split('.')[0]}"

    try:
        aligner = FaceAligner(model_path=landmarker_task)
        detector = AltFreezingVideoDetector(config_path=config_file, weights_path=weights_file, device=device)
        processor = VideoProcessor(aligner=aligner, detector=detector)
        processor.process(video_path=test_video, output_dir=output_folder)
    except Exception as e:
        print(f"An error occurred. Details are {e}")
