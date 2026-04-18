import time
import queue
from concurrent.futures import ThreadPoolExecutor

import settings
from core.challenge_generator import ChallengeGenerator
from core.challenge_timer import ChallengeTimer
from core.decision_logic import DecisionLogic
from core.feedback_overlay import FeedbackOverlay
from core.video_writer import VideoWriter
from core.statistics_writer import StatisticsWriter
from identity.identity_tracker import IdentityTracker
from interactive.action_enum import get_action_name
from interactive.interactive_runner import InteractiveRunner
from passive.passive_runner import PassiveRunner
from preprocessing.preprocessor import Preprocessor
from preprocessing.video_input import EndOfStreamError


class LivenessDetectionEngine:
    def __init__(self, video_input, output_modules, stop_event, web_output=None, modules=None):
        self.video_input = video_input
        self.output_modules = list(output_modules)
        self.stop_event = stop_event
        self.web_output = web_output

        self.preprocessor = Preprocessor()
        self.interactive_runner, self.passive_runner, self.identity_tracker = modules or self.load_modules()

        self.challenge_generator = ChallengeGenerator()
        self.challenge_timer = ChallengeTimer()
        self.challenge_timer.reset(self.challenge_generator.get_current_action())
        self.decision_logic = DecisionLogic()
        self.feedback_overlay = FeedbackOverlay()
        self.statistics_writer = StatisticsWriter()

        self._last_frame_count = None
        self._last_output_frame_count = None
        self._latest_passive_result = None
        self._latest_identity_result = None
        self._latest_deepfake_score = None
        self.final_status = None
        self._video_writer_initialized = False

    @staticmethod
    def load_modules():
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(cls) for cls in (InteractiveRunner, PassiveRunner, IdentityTracker)]
            return tuple(f.result() for f in futures)

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

    def _wait_for_first_frame(self):
        connected = getattr(self.video_input, 'connected', None)
        if connected is None:
            return True
        while not self.stop_event.is_set():
            if connected.wait(timeout=1.0):
                return True
        return False

    def run(self):
        self.video_input.print_video_info()
        self._start_outputs()
        self.video_input.start()

        if self.video_input.is_live:
            if not self._wait_for_first_frame():
                return

        self.passive_runner.start()
        self.identity_tracker.start()

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
                interactive_result = self.interactive_runner.process_frame(preprocessed, current_action, self.challenge_timer)

                preprocessed_passive = self.preprocessor.prepare_passive_input(preprocessed, interactive_result.face_result)
                self.passive_runner.submit(preprocessed_passive)

                self.identity_tracker.submit(preprocessed_passive.get("aligned_face"), frame_count)
                identity_result = self.identity_tracker.get_result()
                if identity_result is not None:
                    self._latest_identity_result = identity_result

                if interactive_result.completed_action is not None:
                    self.decision_logic.complete_action(interactive_result.completed_action, frame_count, self.passive_runner)
                    self.challenge_generator.mark_current_completed()
                    self.challenge_timer.reset(self.challenge_generator.get_current_action())

                decision_action = self.challenge_generator.get_current_action()
                actions_completed = self.challenge_generator.completed_count()
                actions_total = self.challenge_generator.total_actions()
                timeout_failed = self.challenge_timer.failed and not self.challenge_generator.is_finished()
                decision = self._process_frame(frame_count, preprocessed, interactive_result, identity_result, decision_action, actions_completed, actions_total, timeout_failed)
                passive_result = decision.get('passive')
                if decision['status'] in ('pass', 'fail'):
                    self.final_status = decision
                    if self.video_input.is_live:
                        break

                if frame_count % 30 == 0:
                    self._log_progress(frame_count, time.time() - start_time, processed_count, passive_result, identity_result)

                processed_count += 1
        finally:
            self._latest_passive_result = self.passive_runner.get_passive_result()
            final_decision = None
            deepfake_score = None
            if self.final_status:
                final_decision = 'timeout' if self.final_status.get('display_status') == 'Action Timeout' else self.final_status.get('status')
                deepfake_score = self.final_status.get('deepfake_score')

            temporal_window_stats = self.passive_runner.get_temporal_window_stats()
            final_deepfake = deepfake_score if deepfake_score is not None else self._latest_deepfake_score
            summary = self.statistics_writer.write_summary(
                self._latest_passive_result, self._latest_identity_result,
                final_decision, settings.config.deepfake_label, final_deepfake, temporal_window_stats,
            )
            print(f"--- SUMMARY ---\n{summary}")
            self.statistics_writer.close()
            self.video_input.stop()
            self.passive_runner.stop()
            self.identity_tracker.stop()
            self._stop_outputs()
            print(f"Total {self._last_frame_count} frames (interactive processed: {processed_count} frames) in {time.time() - start_time:.1f}s")

    def _log_progress(self, frame_count, elapsed, processed_count, passive_result, identity_result):
        def pct(v): return f"{v * 100:2.0f}%" if v is not None else "N/A"
        fps = processed_count / elapsed if elapsed > 0 else 0
        p_cur = passive_result.score_cur if passive_result else None
        p_avg = passive_result.score_avg if passive_result else None
        p_smooth = passive_result.score_smooth if passive_result else None
        p_s = passive_result.spatial.current_score if passive_result else None
        p_f = passive_result.frequency.current_score if passive_result else None
        p_t = passive_result.temporal.current_score if passive_result else None
        id_score = identity_result.identity_score if identity_result else None
        id_min_sim = identity_result.min_similarity if identity_result else None
        print(
            f"frame={frame_count:04d} fps={fps:2.0f}"
            f" | deepfake_score={pct(self._latest_deepfake_score)} avg={pct(p_avg)} cur={pct(p_cur)} smooth={pct(p_smooth)}"
            f" | spatial={pct(p_s)} frequency={pct(p_f)} temporal={pct(p_t)}"
            f" | identity_score={pct(id_score)} min_similarity={pct(id_min_sim)}"
        )

    def _send_web_overlay(self, interactive_result, passive_result, identity_result, action, decision, completed, total, overlay):
        if not self.web_output:
            return
        result = {
            'type': 'result',
            'action': get_action_name(action),
            'progress': interactive_result.challenge_progress,
            'completed': completed,
            'total': total,
            'status': decision['status'] if decision else 'pending',
            'status_text': decision['display_status'] if decision else f'{completed}/{total} actions completed',
            'face_detected': interactive_result.actions.get('face_detected', False),
            'completed_action': overlay.get('completed_action') if overlay else None,
            'final': decision is not None and decision['status'] in ('pass', 'fail'),
            'passive_score_avg': passive_result.score_avg if passive_result else None,
            'passive_score_smooth': passive_result.score_smooth if passive_result else None,
            'deepfake_score': decision.get('deepfake_score') if decision else None,
            'identity_similarity': identity_result.similarity if identity_result else None,
            'identity_score': identity_result.identity_score if identity_result else None,
        }
        self.web_output.put_overlay(result)

    def _process_frame(self, frame_count, preprocessed, interactive_result, identity_result, action, completed, total, timeout_failed):
        passive_result = self.passive_runner.get_passive_result()
        if passive_result is not None:
            self._latest_passive_result = passive_result
        decision = self.decision_logic.fuse(passive_result, identity_result, completed, total, timeout_failed, passive_runner=self.passive_runner)
        self._latest_deepfake_score = decision.get('deepfake_score')
        passive_delay_frames = frame_count - passive_result.spatial.current_frame if passive_result else 0
        overlay = {
            'current_action': action,
            'challenge_progress': interactive_result.challenge_progress,
            'challenge_completed': completed,
            'challenge_total': total,
            'completed_action': interactive_result.completed_action,
            'decision': decision['status'],
            'decision_text': decision['display_status'],
            'passive_delay': passive_delay_frames,
        }
        rendered = self.feedback_overlay.draw(preprocessed['frame'], interactive_result, passive_result, identity_result, overlay)
        action_message = overlay.get('completed_action')

        if self.video_input.is_live and self._last_output_frame_count is not None:
            frames_skipped = frame_count - self._last_output_frame_count - 1
            if frames_skipped > 0:
                for module in self.output_modules:
                    if isinstance(module, VideoWriter):
                        module.put_repeated_frames(frames_skipped)
        for module in self.output_modules:
            module.put_frame(rendered, frame_count, action_message)

        self._last_output_frame_count = frame_count
        self.statistics_writer.write_frame(frame_count, interactive_result, passive_result, identity_result, action, completed, total)
        if self.web_output:
            self._send_web_overlay(interactive_result, passive_result, identity_result, action, decision, completed, total, overlay)
        return decision

    def _start_outputs(self):
        for module in self.output_modules:
            module.start()

    def _stop_outputs(self):
        for module in reversed(self.output_modules):
            module.stop()
