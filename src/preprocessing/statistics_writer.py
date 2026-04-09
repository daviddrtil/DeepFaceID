import settings
from interactive.action_enum import get_action_name


class StatisticsWriter:
    def __init__(self):
        self.file = open(settings.config.output_stats_path, "w", encoding="utf-8")

    @staticmethod
    def _to_action_text(values):
        if not values:
            return "[]"
        processed_list = [v.value if hasattr(v, "value") else str(v) for v in values]
        return repr(processed_list)

    @staticmethod
    def _to_score_text(value):
        return "None  " if value is None else f"{value:.4f}"

    def write_frame(self, frame_count, interactive_data, passive_state, current_action=None):
        actions = interactive_data.actions
        yaw = actions.get("yaw")
        pitch = actions.get("pitch")
        roll = actions.get("roll")
        yaw_text = "None" if yaw is None else f"{yaw:+6.2f}"
        pitch_text = "None" if pitch is None else f"{pitch:+6.2f}"
        roll_text = "None" if roll is None else f"{roll:+6.2f}"
        pose_text = self._to_action_text(actions.get("pose"))
        occlusion_text = self._to_action_text(actions.get("occlusions"))
        expression_text = self._to_action_text(actions.get("expressions"))

        p_cur, p_avg, p_s, p_f, p_t = "None  ", "None  ", "None  ", "None  ", "None  "
        s_frame, f_frame, t_frame = 0, 0, 0
        if passive_state is not None:
            p_cur = self._to_score_text(passive_state.score_cur)
            p_avg = self._to_score_text(passive_state.score_avg)
            p_s = self._to_score_text(passive_state.spatial.current_score)
            p_f = self._to_score_text(passive_state.frequency.current_score)
            p_t = self._to_score_text(passive_state.temporal.current_score)
            s_frame = passive_state.spatial.current_frame
            f_frame = passive_state.frequency.current_frame
            t_frame = passive_state.temporal.current_frame

        action_text = get_action_name(current_action) or "None"

        self.file.write(
            f"frame={frame_count:04d} | "
            f"spatial_frame={s_frame:04d} frequency_frame={f_frame:04d} temporal_frame={t_frame:04d} "
            f"passive_cur={p_cur} passive_avg={p_avg} "
            f"spatial={p_s} frequency={p_f} temporal={p_t} | "
            f"action={action_text} | "
            f"face={int(actions.get('face_detected', False))} hand={int(actions.get('hand_detected', False))} "
            f"overlap={int(actions.get('hand_face_overlap', False))} yaw={yaw_text} pitch={pitch_text} roll={roll_text} "
            f"pose={pose_text} occlusions={occlusion_text} expressions={expression_text}\n"
        )
        self.file.flush()

    def write_summary(self, passive_state, final_decision=None, deepfake_label=None):
        self.file.write("\n--- SUMMARY ---\n")
        if passive_state:
            s = f"{passive_state.spatial.avg_score:.4f}" if passive_state.spatial.avg_score else "N/A"
            f = f"{passive_state.frequency.avg_score:.4f}" if passive_state.frequency.avg_score else "N/A"
            t = f"{passive_state.temporal.avg_score:.4f}" if passive_state.temporal.avg_score else "N/A"
            self.file.write(
                f"Average passive scores: spatial={s}({passive_state.spatial.total_count}) "
                f"frequency={f}({passive_state.frequency.total_count}) "
                f"temporal={t}({passive_state.temporal.total_count})\n"
            )
        self.file.write(f"label={deepfake_label or 'unknown'}\n")
        self.file.write(f"final_decision={final_decision or 'unknown'}\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
