import cv2
import logging

logger = logging.getLogger(__name__)


class VideoCapture:
    """비디오 파일에서 N초 간격으로 프레임을 추출하는 캡처 모듈."""

    def __init__(self, source: str, frame_interval_seconds: float = 2.0):
        self.source = source
        self.frame_interval_seconds = frame_interval_seconds
        self._cap = None
        self._fps = 0.0
        self._total_frames = 0
        self._skip_frames = 0
        self._current_frame = 0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error("Failed to open video: %s", self.source)
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._skip_frames = max(1, int(self._fps * self.frame_interval_seconds))
        self._current_frame = 0

        logger.info(
            "Opened %s: %.1f fps, %d frames, skip every %d frames (%.1fs interval)",
            self.source, self._fps, self._total_frames,
            self._skip_frames, self.frame_interval_seconds,
        )
        return True

    def read_frame(self):
        """다음 샘플링 프레임을 읽어 (frame, timestamp_seconds) 반환. 끝이면 (None, None)."""
        if self._cap is None or not self._cap.isOpened():
            return None, None

        # grab()으로 프레임을 건너뛰고, 필요한 프레임만 retrieve()로 디코딩
        for _ in range(self._skip_frames - 1):
            if not self._cap.grab():
                return None, None
            self._current_frame += 1

        ret, frame = self._cap.read()
        if not ret:
            return None, None

        self._current_frame += 1
        timestamp = self._current_frame / self._fps if self._fps > 0 else 0.0
        return frame, timestamp

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def progress(self) -> float:
        if self._total_frames <= 0:
            return 0.0
        return min(1.0, self._current_frame / self._total_frames)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()
