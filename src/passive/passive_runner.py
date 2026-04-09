import random
import torch
import time
from dataclasses import dataclass

from passive.analyzer_worker import AnalyzerWorker
from passive.spatial_analyzer.ucf_detector import UCFDetector


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS = {'spatial': 1.0, 'frequency': 0.0, 'temporal': 0.0}


@dataclass
class AnalyzerResult:
    current_score: float = None
    current_frame: int = 0
    avg_score: float = None
    total_count: int = 0


@dataclass
class PassiveResult:
    spatial: AnalyzerResult
    frequency: AnalyzerResult
    temporal: AnalyzerResult
    score_cur: float = None
    score_avg: float = None

    def __post_init__(self):
        self.score_cur = self._weighted('current_score')
        self.score_avg = self._weighted('avg_score')

    def _weighted(self, attr):
        total, weight = 0.0, 0.0
        for key, w in WEIGHTS.items():
            score = getattr(getattr(self, key), attr)
            if score is not None:
                total += score * w
                weight += w
        return total / weight if weight else None


class SpatialAnalyzer:
    def __init__(self):
        self.detector = UCFDetector(DEVICE)

    def run(self, passive_input):
        tensor = passive_input.get("passive_face_input")
        if tensor is None:
            return None
        return self.detector.predict(tensor.to(DEVICE))


class FrequencyAnalyzer:
    def run(self, passive_input):
        fps = random.uniform(100, 200)
        time.sleep(1.0 / fps)
        return random.uniform(0.4, 0.6)


class TemporalAnalyzer:
    def run(self, passive_input):
        fps = random.uniform(50, 100)
        time.sleep(1.0 / fps)
        return random.uniform(0.4, 0.6)


class PassiveRunner:
    def __init__(self):
        self.spatial = AnalyzerWorker(SpatialAnalyzer(), queue_size=4)
        self.frequency = AnalyzerWorker(FrequencyAnalyzer(), queue_size=4)
        self.temporal = AnalyzerWorker(TemporalAnalyzer(), queue_size=16)
        self._cached_state = None

    def start(self):
        self.spatial.start()
        self.frequency.start()
        self.temporal.start()

    def submit(self, passive_input):
        self.spatial.submit(passive_input)
        self.frequency.submit(passive_input)
        self.temporal.submit(passive_input)

    def get_passive_state(self):
        s_score, s_frame, s_avg, s_count = self.spatial.get_all()
        f_score, f_frame, f_avg, f_count = self.frequency.get_all(ref_frame=s_frame)
        t_score, t_frame, t_avg, t_count = self.temporal.get_all(ref_frame=s_frame)
        self._cached_state = PassiveResult(
            spatial=AnalyzerResult(s_score, s_frame, s_avg, s_count),
            frequency=AnalyzerResult(f_score, f_frame, f_avg, f_count),
            temporal=AnalyzerResult(t_score, t_frame, t_avg, t_count),
        )
        return self._cached_state

    def stop(self):
        self.spatial.stop()
        self.frequency.stop()
        self.temporal.stop()

        state = self._cached_state
        if state:
            s = f"{state.spatial.avg_score:.4f}" if state.spatial.avg_score else "N/A"
            f = f"{state.frequency.avg_score:.4f}" if state.frequency.avg_score else "N/A"
            t = f"{state.temporal.avg_score:.4f}" if state.temporal.avg_score else "N/A"
            print(
                f"Average passive scores: "
                f"spatial={s}({state.spatial.total_count}) | "
                f"frequency={f}({state.frequency.total_count}) | "
                f"temporal={t}({state.temporal.total_count})"
            )
