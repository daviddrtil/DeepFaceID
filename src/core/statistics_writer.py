import settings
from interactive.action_enum import get_action_name, get_action_category


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

    def write_frame(self, frame_count, interactive_result, passive_result, identity_result, current_action, challenge_index, challenge_total):
        actions = interactive_result.actions
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
        if passive_result is not None:
            p_cur = self._to_score_text(passive_result.score_cur)
            p_avg = self._to_score_text(passive_result.score_avg)
            p_s = self._to_score_text(passive_result.spatial.current_score)
            p_f = self._to_score_text(passive_result.frequency.current_score)
            p_t = self._to_score_text(passive_result.temporal.current_score)
            s_frame = passive_result.spatial.current_frame
            f_frame = passive_result.frequency.current_frame
            t_frame = passive_result.temporal.current_frame

        id_sim = self._to_score_text(identity_result.similarity if identity_result else None)
        id_avg = self._to_score_text(identity_result.avg_similarity if identity_result else None)
        id_drift = self._to_score_text(identity_result.drift if identity_result else None)
        id_score = self._to_score_text(identity_result.identity_score if identity_result else None)

        action_text = get_action_name(current_action) or "None"
        action_text = '\'' + action_text + '\''
        category_text = get_action_category(current_action) or "None"
        category_text = '\'' + category_text + '\''

        self.file.write(
            f"frame={frame_count:04d} | "
            f"spatial_frame={s_frame:04d} frequency_frame={f_frame:04d} temporal_frame={t_frame:04d} "
            f"passive_cur={p_cur} passive_avg={p_avg} "
            f"spatial={p_s} frequency={p_f} temporal={p_t} | "
            f"id_sim={id_sim} id_avg={id_avg} id_drift={id_drift} id_score={id_score} | "
            f"face={int(actions.get('face_detected', False))} hand={int(actions.get('hand_detected', False))} "
            f"overlap={int(actions.get('hand_face_overlap', False))} yaw={yaw_text} pitch={pitch_text} roll={roll_text} | "
            f"challenge={challenge_index}/{challenge_total} action_category={category_text} action={action_text} | "
            f"pose={pose_text} occlusions={occlusion_text} expressions={expression_text}\n"
        )
        self.file.flush()

    @staticmethod
    def format_summary(passive_result, identity_result, final_decision, deepfake_label):
        lines = []
        if passive_result:
            s = f"{passive_result.spatial.avg_score:.4f}" if passive_result.spatial.avg_score else "N/A"
            f = f"{passive_result.frequency.avg_score:.4f}" if passive_result.frequency.avg_score else "N/A"
            t = f"{passive_result.temporal.avg_score:.4f}" if passive_result.temporal.avg_score else "N/A"
            lines.append(
                f"Average passive scores: spatial={s}({passive_result.spatial.total_count}) "
                f"frequency={f}({passive_result.frequency.total_count}) "
                f"temporal={t}({passive_result.temporal.total_count})"
            )
        if identity_result:
            lines.append(
                f"Identity: avg_similarity={identity_result.avg_similarity:.4f} "
                f"min_similarity={identity_result.min_similarity:.4f} "
                f"drift={identity_result.drift:.4f} "
                f"identity_score={identity_result.identity_score:.4f} "
                f"embeddings={identity_result.embedding_count}"
            )
        lines.append(f"label={deepfake_label or 'unknown'}")
        lines.append(f"final_decision={final_decision or 'unknown'}")
        return "\n".join(lines)

    def write_summary(self, passive_result, identity_result, final_decision, deepfake_label):
        self.file.write("\n--- SUMMARY ---\n")
        self.file.write(self.format_summary(passive_result, identity_result, final_decision, deepfake_label))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
