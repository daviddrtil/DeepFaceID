import settings


class StatisticsWriter:
    def __init__(self):
        self.file = open(settings.config.output_stats_path, "w", encoding="utf-8")

    @staticmethod
    def _to_action_text(values):
        if not values:
            return "[]"
        processed_list = [v.value if hasattr(v, "value") else str(v) for v in values]
        return repr(processed_list)

    def write_frame(self, frame_count, interactive_data, passive_result):
        actions = interactive_data.get("actions", {})
        yaw = actions.get("yaw")
        pitch = actions.get("pitch")
        roll = actions.get("roll")
        yaw_text = "None" if yaw is None else f"{yaw:+6.2f}"
        pitch_text = "None" if pitch is None else f"{pitch:+6.2f}"
        roll_text = "None" if roll is None else f"{roll:+6.2f}"
        pose_text = self._to_action_text(actions.get("pose"))
        occlusion_text = self._to_action_text(actions.get("occlusions"))
        expression_text = self._to_action_text(actions.get("expressions"))

        passive_score_raw = "None  "
        passive_score_avg = "None  "
        passive_spatial = "None  "
        passive_frequency = "None  "
        passive_temporal = "None  "
        passive_frame = 0
        if passive_result is not None:
            passive_score_raw = f"{passive_result['score_raw']:.4f}"
            passive_score_avg = f"{passive_result['score_avg']:.4f}"
            passive_spatial = f"{passive_result['spatial']:.4f}"
            passive_frequency = f"{passive_result['frequency']:.4f}"
            passive_temporal = f"{passive_result['temporal']:.4f}"
            passive_frame = passive_result.get("frame_count")

        self.file.write(
            f"frame={frame_count:04d} | passive_frame={passive_frame:04d} "
            f"passive_raw={passive_score_raw} passive_avg={passive_score_avg} spatial={passive_spatial} frequency={passive_frequency} temporal={passive_temporal} |"
            f"face={int(actions.get('face_detected', False))} hand={int(actions.get('hand_detected', False))} "
            f"overlap={int(actions.get('hand_face_overlap', False))} yaw={yaw_text} pitch={pitch_text} roll={roll_text} "
            f"pose={pose_text} occlusions={occlusion_text} expressions={expression_text}\n"
        )
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
