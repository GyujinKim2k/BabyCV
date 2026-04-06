import logging

logger = logging.getLogger(__name__)


class FrameSampler:
    """모션 점수 기반 프레임 샘플링 전략. API 호출 빈도를 제어."""

    def __init__(self, motion_threshold: float, api_interval_seconds: float = 30.0):
        """
        Args:
            motion_threshold: 이 값 이상의 모션이 있어야 API 호출 대상
            api_interval_seconds: API 호출 최소 간격 (초)
        """
        self._motion_threshold = motion_threshold
        self._api_interval = api_interval_seconds
        self._last_api_time: float | None = None

    def should_call_api(self, motion_score: float, timestamp: float) -> bool:
        """모션 점수와 타임스탬프를 보고, Gemini API를 호출해야 하는지 판단."""
        # 모션이 임계값 미만이면 호출 불필요
        if motion_score < self._motion_threshold:
            return False

        # API 호출 간격 체크
        if self._last_api_time is not None:
            elapsed = timestamp - self._last_api_time
            if elapsed < self._api_interval:
                return False

        return True

    def mark_api_called(self, timestamp: float):
        """API 호출 완료 후 마지막 호출 시간 갱신."""
        self._last_api_time = timestamp
