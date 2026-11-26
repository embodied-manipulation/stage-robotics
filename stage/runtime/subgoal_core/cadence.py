"""Physical-action observation cadence and actual Planner-delivery gating.

Uses public result/image interfaces; no backend, model, or simulator imports.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import uuid


class ReviewRequired(RuntimeError):
    """The action window is spent and a fresh Planner observation is required."""
    pass


def contains_image(value):
    if isinstance(value, dict):
        if value.get("type") in {"image", "input_image", "image_url"}:
            return True
        return any(contains_image(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_image(item) for item in value)
    return False


class ReviewWindow:
    def __init__(self, actions=40, log_path=None, *, image_window=None):
        if type(actions) is not int or actions < 1:
            raise ValueError("review_window_actions must be a positive integer")
        self.image_window = image_window
        self.cap = actions
        self.used = 0
        self.total = 0
        self.last_capture = -1
        self.last_grant = -1
        self.token = None
        self.active = 0
        self.lock = threading.RLock()
        self.log_path = Path(log_path) if log_path else None

    @property
    def remaining(self):
        with self.lock:
            return self.cap - self.used

    def record(self, event, **details):
        if self.log_path:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps({"event": event, "actions": self.total,
                                         "used": self.used, **details}) + "\n")

    @contextmanager
    def motion(self):
        with self.lock:
            self.active += 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1

    def require(self, actions=1):
        with self.lock:
            if actions > self.cap - self.used:
                raise ReviewRequired("Review the latest observation before further motion")

    def run_actions(self, action_count, function, *args, **kwargs):
        # Reserve before the RPC. On an RPC error do not refund uncertain motion.
        with self.lock:
            self.require(action_count)
            self.used += action_count
            self.total += action_count
        return function(*args, **kwargs)

    def observe(self, result, tool_name):
        captured = bool(result.images)
        images = self.image_window
        if images is not None:
            result = images.observe(result, self.total)
            captured = (captured and bool(images.frames)
                        and images.frames[-1]['action'] == self.total
                        and images.frames[-1]['artifact_step'] == result.data.get('step'))
        with self.lock:
            # A text-only or failed capture must not unlock another motion window.
            if captured and self.total > self.last_capture:
                self.last_capture = self.total
                self.token = "review-" + uuid.uuid4().hex
            status = {
                "limit_actions": self.cap,
                "action_count": self.total, "review_token": self.token,
                "semantics": "maximum_actions_between_observed_planner_decisions",
            }
            result.data["review_window"] = status
            self.record("tool_return", tool=tool_name, **status)
        return result

    def planner_request(self, payload):
        with self.lock:
            # Requires both a newly captured image and its actual tool-result
            # token in the next model input. A retry of an old input cannot reset.
            delivered = json.dumps(payload.get("input", []), ensure_ascii=False)
            if (self.active == 0 and self.last_capture > self.last_grant and self.token
                    and self.token in delivered and contains_image(payload.get("input", []))):
                previous = self.used
                self.used = 0
                self.last_grant = self.last_capture
                self.record("planner_review", previous_window_actions=previous,
                            review_token=self.token)
