class ChallengeTimer:
    def __init__(self, hold_duration_seconds=1.0, fail_timeout_seconds=15.0):
        self.default_hold_duration_ms = int(hold_duration_seconds * 1000)
        self.hold_duration_ms = self.default_hold_duration_ms
        self.hold_start_ms = None
        self.fail_timeout_ms = int(fail_timeout_seconds * 1000)
        self.action_start_ms = None
        self.failed = False

    def reset(self, action=None):
        self.hold_start_ms = None
        self.action_start_ms = None
        self.failed = False
        if hasattr(action, 'duration_seconds'):
            self.hold_duration_ms = int(action.duration_seconds * 1000)
        else:
            self.hold_duration_ms = self.default_hold_duration_ms

    def update(self, matched, timestamp_ms):
        current_ms = int(timestamp_ms)
        if self.action_start_ms is None:
            self.action_start_ms = current_ms

        if self.fail_timeout_ms > 0 and (current_ms - self.action_start_ms) >= self.fail_timeout_ms:
            self.hold_start_ms = None
            self.failed = True
            return False, 0.0

        if not matched:
            self.hold_start_ms = None
            return False, 0.0

        if self.hold_start_ms is None:
            self.hold_start_ms = current_ms

        elapsed_ms = current_ms - self.hold_start_ms
        if self.hold_duration_ms <= 0:
            return True, 1.0

        progress = min(1.0, max(0.0, elapsed_ms / self.hold_duration_ms))
        return elapsed_ms >= self.hold_duration_ms, progress
