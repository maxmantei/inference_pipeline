import threading


class OrStopEvent(threading.Event):
    """Read-only event that reports set when either source event is set."""

    def __init__(self, left: threading.Event, right: threading.Event) -> None:
        super().__init__()
        self._left = left
        self._right = right

    def is_set(self) -> bool:
        return self._left.is_set() or self._right.is_set()
