import numpy as np
import pytest

from src.motion_detector import MotionDetector


class TestMotionDetector:
    def test_no_motion_returns_low_score(self):
        """동일한 프레임을 연속 입력하면 모션 점수가 낮아야 함."""
        detector = MotionDetector()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        # 배경 모델 안정화 (처음 몇 프레임은 높은 점수가 나올 수 있음)
        for _ in range(30):
            detector.detect(frame)

        result = detector.detect(frame)
        assert result["motion_score"] < 5.0

    def test_motion_detected_on_change(self):
        """프레임에 큰 변화가 있으면 모션 점수가 높아야 함."""
        detector = MotionDetector()

        # 배경 안정화: 어두운 프레임
        dark_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        for _ in range(30):
            detector.detect(dark_frame)

        # 밝은 프레임 입력 → 큰 변화
        bright_frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        result = detector.detect(bright_frame)
        assert result["motion_score"] > 10.0

    def test_roi_crops_frame(self):
        """ROI 설정 시 해당 영역만 분석해야 함."""
        roi = [100, 100, 200, 200]
        detector = MotionDetector(roi=roi)

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        # 배경 안정화
        for _ in range(30):
            detector.detect(frame)

        # ROI 영역만 변경
        modified = frame.copy()
        modified[100:300, 100:300] = 255
        result = detector.detect(modified)
        # ROI 내 변화가 있으므로 모션 감지
        assert result["motion_score"] > 0

    def test_roi_ignores_outside(self):
        """ROI 밖의 변화는 무시해야 함."""
        roi = [100, 100, 100, 100]  # 작은 영역
        detector = MotionDetector(roi=roi)

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(30):
            detector.detect(frame)

        # ROI 밖에서만 변경
        modified = frame.copy()
        modified[0:50, 0:50] = 255  # ROI 밖
        result = detector.detect(modified)
        assert result["motion_score"] < 5.0

    def test_result_has_required_keys(self):
        """결과에 필수 키가 포함되어야 함."""
        detector = MotionDetector()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = detector.detect(frame)
        assert "motion_score" in result
        assert "presence_score" in result
        assert "foreground_mask" in result
