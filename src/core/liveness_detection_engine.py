from collections import deque
import threading
import time
import queue

import settings
from core.challenge_generator import ChallengeGenerator
from core.challenge_timer import ChallengeTimer
from core.decision_logic import DecisionLogic
from core.feedback_overlay import FeedbackOverlay
from interactive.interactive_runner import InteractiveRunner
from passive.passive_runner import PassiveRunner
from preprocessing.preprocessor import Preprocessor
from preprocessing.video_writer import VideoWriter
from preprocessing.statistics_writer import StatisticsWriter
from preprocessing.video_input import EndOfStreamError


class LivenessDetectionEngine:
    def __init__(self, video_input, output_modules, stop_event, web_output=None):
        self.video_input = video_input
        self.output_modules = list(output_modules)
        self.stop_event = stop_event
        self.web_output = web_output

        self.preprocessor = Preprocessor()
        self.interactive_runner = InteractiveRunner()
        self.passive_runner = PassiveRunner()
        self.challenge_generator = ChallengeGenerator()
        self.challenge_timer = ChallengeTimer()
        self.decision_logic = DecisionLogic()
        self.feedback_overlay = FeedbackOverlay()
        self.statistics_writer = StatisticsWriter()

        self._last_frame_count = None
        self.render_buffer = deque()
        self.delay_frames = 5
        self.final_status = None
        self._video_writer_initialized = False

    def run(self):
        self.video_input.print_video_info()
        self._start_outputs()
        self.video_input.start()
        self.passive_runner.start()

        start_time = time.time()
        processed_count = 0

        try:
            while not self.stop_event.is_set():
                if settings.config.max_frames is not None and processed_count >= settings.config.max_frames:
                    break

                try:
                    frame, timestamp_ms, frame_count = self.video_input.get_frame()
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                except EndOfStreamError:
                    break

                self._init_video_writer(frame)
                preprocessed = self.preprocessor.process_frame(frame, timestamp_ms, frame_count, self.video_input.fps)
                
                self._last_frame_count = frame_count

                current_action = self.challenge_generator.get_current_action()
                interactive_data = self.interactive_runner.process_frame(preprocessed, current_action, self.challenge_timer)

                preprocessed_passive = self.preprocessor.prepare_passive_input(preprocessed, interactive_data["face_result"])
                if preprocessed_passive.get("passive_face_input") is not None:
                    self.passive_runner.submit(preprocessed_passive)

                if interactive_data['completed_action'] is not None:
                    self.challenge_generator.mark_current_completed()
                    self.challenge_timer.reset()

                self.render_buffer.append({
                    'frame_count': frame_count,
                    'preprocessed': preprocessed,
                    'interactive_data': interactive_data,
                    'current_action': self.challenge_generator.get_current_action(),
                    'actions_completed_count': self.challenge_generator.completed_count(),
                    'actions_count': self.challenge_generator.total_actions(),
                    'timeout_failed': self.challenge_timer.failed and not self.challenge_generator.is_finished()
                })

                if len(self.render_buffer) < self.delay_frames:
                    # delayed rendering to synchronize passive module
                    if self.web_output and self.video_input.is_live:
                        self._send_web_overlay(interactive_data, current_action, None)
                    continue

                delayed_data = self.render_buffer.popleft()
                d_frame_count = delayed_data['frame_count']
                d_preprocessed = delayed_data['preprocessed']
                d_interactive = delayed_data['interactive_data']
                d_action = delayed_data['current_action']
                d_completed = delayed_data['actions_completed_count']
                d_total = delayed_data['actions_count']
                d_timeout = delayed_data['timeout_failed']

                passive_result = self.passive_runner.get_latest_result()

                passive_score_raw = None
                passive_score_avg = None
                passive_delay_frames = 0
                if passive_result is not None:
                    passive_score_raw = passive_result["score_raw"]
                    passive_score_avg = passive_result["score_avg"]
                    passive_delay_frames = d_frame_count - passive_result["frame_count"]

                decision = self.decision_logic.fuse(
                    d_interactive['actions'],
                    passive_score_avg,
                    d_action,
                    d_completed,
                    d_total,
                    d_timeout,
                )

                overlay = {
                    'current_action': d_action.value if d_action is not None else 'Completed',
                    'challenge_progress': d_interactive['challenge_progress'],
                    'challenge_completed': d_completed,
                    'challenge_total': d_total,
                    'completed_action': d_interactive['completed_action'],
                    'decision': decision['status'],
                    'decision_text': decision['display_status'],
                    'passive_delay': passive_delay_frames,
                }

                rendered = self.feedback_overlay.draw(
                    d_preprocessed['frame'],
                    d_interactive['face_result'],
                    d_interactive['hand_result'],
                    d_interactive['actions'],
                    d_interactive['hand_mask'],
                    passive_score_raw,
                    overlay,
                )

                action_message = overlay.get('completed_action')
                for module in self.output_modules:
                    module.put_frame(rendered, d_frame_count, action_message)

                self.statistics_writer.write_frame(d_frame_count, d_interactive, passive_result)

                if self.web_output:
                    self._send_web_overlay(d_interactive, d_action, decision, d_completed, d_total, overlay)

                if decision['status'] in ('pass', 'fail'):
                    self.final_status = decision
                    if self.video_input.is_live:
                        break

                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    current_fps = processed_count / elapsed if elapsed > 0 else 0
                    passive_raw_pct = f"{passive_score_raw * 100:2.0f}%" if passive_score_raw is not None else "N/A"
                    passive_avg_pct = f"{passive_score_avg * 100:2.0f}%" if passive_score_avg is not None else "N/A"
                    print(f"frame={processed_count:04d} fps={current_fps:2.0f} passive_raw={passive_raw_pct} passive_avg={passive_avg_pct} delay={passive_delay_frames}")

                processed_count += 1
        finally:
            self.statistics_writer.close()
            self.video_input.stop()
            self.passive_runner.stop()
            self.interactive_runner.stop()
            self._stop_outputs()
            print(f"Processed {processed_count} frames in {time.time() - start_time:.1f} seconds.")

    def _send_web_overlay(self, interactive_data, action, decision, completed=0, total=0, overlay=None):
        if not self.web_output:
            return
        
        result = {
            'type': 'result',
            'action': action.value if action else None,
            'progress': interactive_data['challenge_progress'],
            'completed': completed,
            'total': total,
            'status': decision['status'] if decision else 'pending',
            'status_text': decision['display_status'] if decision else f'{completed}/{total} actions completed',
            'face_detected': interactive_data['actions'].get('face_detected', False),
            'completed_action': overlay.get('completed_action') if overlay else None,
            'final': decision is not None and decision['status'] in ('pass', 'fail')
        }
        self.web_output.put_overlay(result)

    def _start_outputs(self):
        for module in self.output_modules:
            module.start()

    def _stop_outputs(self):
        for module in reversed(self.output_modules):
            module.stop()

    def _init_video_writer(self, frame):
        if self._video_writer_initialized or not self.video_input.is_live:
            return
        if not settings.config.save_output:
            self._video_writer_initialized = True
            return
        height, width = frame.shape[:2]
        writer = VideoWriter(
            width=width,
            height=height,
            fps=self.video_input.fps,
        )
        writer.start()
        self.output_modules.append(writer)
        self._video_writer_initialized = True
        print(f"VideoWriter initialized: {width}x{height} at {self.video_input.fps:.0f} fps")
