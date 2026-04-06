import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SleepState(Enum):
    AWAKE = "awake"
    FALLING_ASLEEP = "falling_asleep"
    ASLEEP = "asleep"
    WAKING_UP = "waking_up"


class ActivityState(Enum):
    IDLE = "idle"
    EATING = "eating"
    DIAPER_CHANGE = "diaper_change"


@dataclass
class Event:
    timestamp: float
    camera: str
    event_type: str  # sleep_start, sleep_end, eating_start, eating_end, ...
    confidence: float = 0.0
    notes: str = ""


@dataclass
class BedroomTracker:
    """침실 상태 머신: 모션 + 존재감 기반 수면/기상 감지."""

    motion_threshold: float = 5.0
    presence_threshold_minutes: float = 3.0

    state: SleepState = SleepState.AWAKE
    _state_start_time: float = 0.0
    _low_motion_start: float | None = None
    _high_motion_start: float | None = None
    _absence_start: float | None = None

    # 디바운싱 파라미터 (초)
    FALLING_ASLEEP_DURATION: float = 300.0  # 5분
    WAKING_UP_DURATION: float = 120.0       # 2분

    def update(self, motion_score: float, presence_score: float, timestamp: float) -> Event | None:
        """
        모션/존재감 점수를 받아 상태를 업데이트하고, 이벤트가 발생하면 반환.
        """
        event = None
        is_low_motion = motion_score < self.motion_threshold
        is_absent = presence_score < self.motion_threshold  # 침대에 아기 없음

        if self.state == SleepState.AWAKE:
            if is_low_motion and not is_absent:
                # 아기가 침대에 있고 조용해짐 → 잠들기 시작
                if self._low_motion_start is None:
                    self._low_motion_start = timestamp
                elif timestamp - self._low_motion_start >= self.FALLING_ASLEEP_DURATION:
                    self.state = SleepState.ASLEEP
                    self._state_start_time = timestamp
                    self._low_motion_start = None
                    event = Event(
                        timestamp=timestamp - self.FALLING_ASLEEP_DURATION,
                        camera="bedroom",
                        event_type="sleep_start",
                        confidence=0.8,
                        notes=f"motion={motion_score:.1f}, presence={presence_score:.1f}",
                    )
                    logger.info("[%.0fs] 수면 시작 감지", timestamp)
            else:
                self._low_motion_start = None

        elif self.state == SleepState.ASLEEP:
            if not is_low_motion or is_absent:
                # 움직임 증가 또는 아기 부재 → 기상 중
                self.state = SleepState.WAKING_UP
                self._high_motion_start = timestamp
                self._absence_start = timestamp if is_absent else None
            # 여전히 잠들어 있으면 아무것도 하지 않음

        elif self.state == SleepState.WAKING_UP:
            if is_absent:
                # 아기가 침대에서 사라진 상태 추적
                if self._absence_start is None:
                    self._absence_start = timestamp
                absence_duration = timestamp - self._absence_start
                if absence_duration >= self.presence_threshold_minutes * 60:
                    # N분 이상 부재 → 기상 확정
                    self.state = SleepState.AWAKE
                    self._state_start_time = timestamp
                    self._high_motion_start = None
                    self._absence_start = None
                    event = Event(
                        timestamp=self._absence_start,
                        camera="bedroom",
                        event_type="sleep_end",
                        confidence=0.9,
                        notes=f"absence {absence_duration:.0f}s, baby left bed",
                    )
                    logger.info("[%.0fs] 기상 감지 (침대 이탈)", timestamp)
            elif not is_low_motion:
                # 활발한 움직임 지속
                self._absence_start = None
                if self._high_motion_start is None:
                    self._high_motion_start = timestamp
                elif timestamp - self._high_motion_start >= self.WAKING_UP_DURATION:
                    # 2분 이상 높은 움직임 → 기상 확정
                    self.state = SleepState.AWAKE
                    self._state_start_time = timestamp
                    event = Event(
                        timestamp=self._high_motion_start,
                        camera="bedroom",
                        event_type="sleep_end",
                        confidence=0.7,
                        notes=f"motion={motion_score:.1f}, sustained high motion",
                    )
                    logger.info("[%.0fs] 기상 감지 (활발한 움직임)", timestamp)
                    self._high_motion_start = None
            else:
                # 다시 조용해짐 → 뒤척임이었음, 다시 수면
                self.state = SleepState.ASLEEP
                self._high_motion_start = None
                self._absence_start = None
                logger.debug("[%.0fs] 뒤척임 판정 → 수면 복귀", timestamp)

        return event


@dataclass
class LivingRoomTracker:
    """거실 상태 머신: Gemini Flash 분류 기반 활동 감지."""

    state: ActivityState = ActivityState.IDLE
    _state_start_time: float = 0.0
    _consecutive_counts: dict = field(default_factory=lambda: {
        "eating": 0, "diaper_change": 0, "idle": 0,
    })

    # 디바운싱: 연속 N회 확인 필요
    EATING_START_THRESHOLD: int = 2
    DIAPER_START_THRESHOLD: int = 2
    EATING_END_THRESHOLD: int = 3
    DIAPER_END_THRESHOLD: int = 2

    def update(self, classification: dict, timestamp: float) -> Event | None:
        """
        Gemini 분류 결과를 받아 상태 업데이트. 이벤트 발생 시 반환.
        """
        activity = classification.get("activity", "none")
        confidence = classification.get("confidence", 0.0)
        event = None

        # 연속 카운트 업데이트
        for key in self._consecutive_counts:
            if key == activity:
                self._consecutive_counts[key] += 1
            elif key == "idle" and activity in ("none", "other", "playing"):
                self._consecutive_counts[key] += 1
            else:
                self._consecutive_counts[key] = 0

        if self.state == ActivityState.IDLE:
            if self._consecutive_counts["eating"] >= self.EATING_START_THRESHOLD:
                self.state = ActivityState.EATING
                self._state_start_time = timestamp
                self._reset_counts()
                event = Event(
                    timestamp=timestamp,
                    camera="livingroom",
                    event_type="eating_start",
                    confidence=confidence,
                    notes=classification.get("details", ""),
                )
                logger.info("[%.0fs] 식사 시작 감지", timestamp)

            elif self._consecutive_counts["diaper_change"] >= self.DIAPER_START_THRESHOLD:
                self.state = ActivityState.DIAPER_CHANGE
                self._state_start_time = timestamp
                self._reset_counts()
                event = Event(
                    timestamp=timestamp,
                    camera="livingroom",
                    event_type="diaper_change_start",
                    confidence=confidence,
                    notes=classification.get("details", ""),
                )
                logger.info("[%.0fs] 기저귀 교체 시작 감지", timestamp)

        elif self.state == ActivityState.EATING:
            if self._consecutive_counts["idle"] >= self.EATING_END_THRESHOLD:
                self.state = ActivityState.IDLE
                duration = timestamp - self._state_start_time
                self._reset_counts()
                event = Event(
                    timestamp=timestamp,
                    camera="livingroom",
                    event_type="eating_end",
                    confidence=confidence,
                    notes=f"duration={duration:.0f}s",
                )
                logger.info("[%.0fs] 식사 종료 감지 (%.0f초)", timestamp, duration)

        elif self.state == ActivityState.DIAPER_CHANGE:
            if self._consecutive_counts["idle"] >= self.DIAPER_END_THRESHOLD:
                self.state = ActivityState.IDLE
                duration = timestamp - self._state_start_time
                self._reset_counts()
                event = Event(
                    timestamp=timestamp,
                    camera="livingroom",
                    event_type="diaper_change_end",
                    confidence=confidence,
                    notes=f"duration={duration:.0f}s",
                )
                logger.info("[%.0fs] 기저귀 교체 종료 감지 (%.0f초)", timestamp, duration)

        return event

    def _reset_counts(self):
        for key in self._consecutive_counts:
            self._consecutive_counts[key] = 0
