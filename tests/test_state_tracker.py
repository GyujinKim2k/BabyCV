import pytest

from src.state_tracker import (
    BedroomTracker, LivingRoomTracker,
    SleepState, ActivityState,
)


class TestBedroomTracker:
    def test_initial_state_is_awake(self):
        tracker = BedroomTracker()
        assert tracker.state == SleepState.AWAKE

    def test_falls_asleep_after_sustained_low_motion(self):
        """5분 이상 낮은 모션 → 수면 시작."""
        tracker = BedroomTracker(motion_threshold=5)
        events = []

        # 5분(300초) 동안 2초 간격으로 낮은 모션 입력
        for t in range(0, 310, 2):
            event = tracker.update(motion_score=1.0, presence_score=30.0, timestamp=float(t))
            if event:
                events.append(event)

        assert tracker.state == SleepState.ASLEEP
        assert len(events) == 1
        assert events[0].event_type == "sleep_start"

    def test_does_not_sleep_without_presence(self):
        """침대에 아기가 없으면(낮은 존재감) 수면 시작 안 됨."""
        tracker = BedroomTracker(motion_threshold=5)
        events = []

        for t in range(0, 400, 2):
            event = tracker.update(motion_score=1.0, presence_score=1.0, timestamp=float(t))
            if event:
                events.append(event)

        assert tracker.state == SleepState.AWAKE
        assert len(events) == 0

    def test_wakes_up_on_absence(self):
        """수면 중 침대에서 아기가 사라지면 기상."""
        tracker = BedroomTracker(motion_threshold=5, presence_threshold_minutes=3)

        # 먼저 수면 상태로 만들기
        for t in range(0, 310, 2):
            tracker.update(motion_score=1.0, presence_score=30.0, timestamp=float(t))
        assert tracker.state == SleepState.ASLEEP

        # 모션 증가 → WAKING_UP 전환
        tracker.update(motion_score=20.0, presence_score=30.0, timestamp=320.0)
        assert tracker.state == SleepState.WAKING_UP

        # 3분간 부재 → 기상 확정
        events = []
        for t in range(322, 510, 2):
            event = tracker.update(motion_score=0.0, presence_score=1.0, timestamp=float(t))
            if event:
                events.append(event)

        assert tracker.state == SleepState.AWAKE
        assert any(e.event_type == "sleep_end" for e in events)

    def test_tossing_returns_to_sleep(self):
        """뒤척임(잠깐 움직임 후 다시 조용) → 수면 복귀."""
        tracker = BedroomTracker(motion_threshold=5)

        # 수면 상태로 만들기
        for t in range(0, 310, 2):
            tracker.update(motion_score=1.0, presence_score=30.0, timestamp=float(t))
        assert tracker.state == SleepState.ASLEEP

        # 잠깐 움직임
        tracker.update(motion_score=20.0, presence_score=30.0, timestamp=320.0)
        assert tracker.state == SleepState.WAKING_UP

        # 바로 다시 조용해짐 → 수면 복귀
        tracker.update(motion_score=1.0, presence_score=30.0, timestamp=322.0)
        assert tracker.state == SleepState.ASLEEP


class TestLivingRoomTracker:
    def test_initial_state_is_idle(self):
        tracker = LivingRoomTracker()
        assert tracker.state == ActivityState.IDLE

    def test_eating_detected_after_consecutive(self):
        """연속 2회 eating 분류 → 식사 시작."""
        tracker = LivingRoomTracker()

        e1 = tracker.update({"activity": "eating", "confidence": 0.9}, 0.0)
        assert e1 is None  # 1회는 부족

        e2 = tracker.update({"activity": "eating", "confidence": 0.9}, 30.0)
        assert e2 is not None
        assert e2.event_type == "eating_start"
        assert tracker.state == ActivityState.EATING

    def test_eating_ends_after_consecutive_idle(self):
        """식사 중 연속 3회 비-eating → 식사 종료."""
        tracker = LivingRoomTracker()

        # 식사 시작
        tracker.update({"activity": "eating", "confidence": 0.9}, 0.0)
        tracker.update({"activity": "eating", "confidence": 0.9}, 30.0)
        assert tracker.state == ActivityState.EATING

        # 3회 연속 idle
        tracker.update({"activity": "playing", "confidence": 0.8}, 60.0)
        tracker.update({"activity": "none", "confidence": 0.5}, 90.0)
        e = tracker.update({"activity": "other", "confidence": 0.5}, 120.0)

        assert e is not None
        assert e.event_type == "eating_end"
        assert tracker.state == ActivityState.IDLE

    def test_diaper_change_detected(self):
        """연속 2회 diaper_change 분류 → 기저귀 교체 시작."""
        tracker = LivingRoomTracker()

        tracker.update({"activity": "diaper_change", "confidence": 0.8}, 0.0)
        e = tracker.update({"activity": "diaper_change", "confidence": 0.8}, 30.0)

        assert e is not None
        assert e.event_type == "diaper_change_start"
        assert tracker.state == ActivityState.DIAPER_CHANGE

    def test_single_classification_not_enough(self):
        """1회만 분류되면 상태 전환 안 됨 (디바운싱)."""
        tracker = LivingRoomTracker()

        tracker.update({"activity": "eating", "confidence": 0.9}, 0.0)
        tracker.update({"activity": "none", "confidence": 0.5}, 30.0)  # 중간에 끊김

        assert tracker.state == ActivityState.IDLE
