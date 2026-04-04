from collections import deque
import random
import torch
import time

from passive.analyzer_worker import AnalyzerWorker
from passive.spatial_analyzer.ucf_detector import UCFDetector
from passive.passive_score_aggregator import PassiveScoreAggregator


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEEPFAKE_SCORE_THRESHOLD = 0.50
SMOOTHING_WINDOW = 5


class SpatialAnalyzer:
    def __init__(self):
        self.detector = UCFDetector(DEVICE)
        self.recent_scores = deque(maxlen=SMOOTHING_WINDOW)

    def run(self, passive_input):
        input_tensor = passive_input.get("passive_face_input")
        if input_tensor is None:
            return None
        raw_score = self.detector.predict(input_tensor.to(DEVICE))
        return raw_score
        # self.recent_scores.append(float(raw_score))   # TODO: consider using SMOOTHING_WINDOW
        # return float(sum(self.recent_scores) / len(self.recent_scores))


class FrequencyAnalyzer:
    def run(self, passive_input):
        # fps = random.uniform(10, 20)
        # time.sleep(1.0 / fps)
        return random.uniform(0.4, 0.6)


class TemporalAnalyzer:
    def run(self, passive_input):
        # fps = random.uniform(5, 10)
        # time.sleep(1.0 / fps)
        return random.uniform(0.4, 0.6)


class PassiveRunner:
    def __init__(self):
        self.spatial_worker = AnalyzerWorker(SpatialAnalyzer(), queue_size=4)
        self.frequency_worker = AnalyzerWorker(FrequencyAnalyzer(), queue_size=4)
        self.temporal_worker = AnalyzerWorker(TemporalAnalyzer(), queue_size=16)
        self.score_aggregator = PassiveScoreAggregator()

        self.latest_result = None
        self.last_frame_counts = (None, None, None)

        self.score_count = 0
        self.sum_spatial_score = 0.0
        self.sum_frequency_score = 0.0
        self.sum_temporal_score = 0.0
        self.sum_aggregated_score = 0.0

    def start(self):
        self.spatial_worker.start()
        self.frequency_worker.start()
        self.temporal_worker.start()

    def submit(self, passive_input):
        self.spatial_worker.submit(passive_input)
        self.frequency_worker.submit(passive_input)
        self.temporal_worker.submit(passive_input)

    def get_latest_result(self):
        results = (
            self.spatial_worker.get_latest(),
            self.frequency_worker.get_latest(),
            self.temporal_worker.get_latest()
        )
        if None in results:
            return self.latest_result

        current_counts = tuple(result["frame_count"] for result in results)
        if current_counts == self.last_frame_counts:
            return self.latest_result

        s, f, t = (result["score"] for result in results)
        self.sum_spatial_score += s
        self.sum_frequency_score += f
        self.sum_temporal_score += t
        self.last_frame_counts = current_counts
        self.score_count += 1

        score_raw = self.score_aggregator.run(s, f, t)
        self.sum_aggregated_score += score_raw
        score_avg = self.sum_aggregated_score / self.score_count

        self.latest_result = {
            "score_raw": score_raw,
            "score_avg": score_avg,
            "spatial": s,
            "frequency": f,
            "temporal": t,
            "frame_count": current_counts[0]
        }
        return self.latest_result

    def stop(self):
        self.spatial_worker.stop()
        self.frequency_worker.stop()
        self.temporal_worker.stop()

        if not self.score_count:
            print("Average passive scores: spatial=N/A frequency=N/A temporal=N/A")
            return

        print(
            f"Average passive scores: "
            f"spatial={self.sum_spatial_score / self.score_count:.4f} "
            f"frequency={self.sum_frequency_score / self.score_count:.4f} "
            f"temporal={self.sum_temporal_score / self.score_count:.4f}"
        )
