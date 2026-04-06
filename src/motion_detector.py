import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MotionDetector:
    """OpenCV 배경 차분(MOG2) 기반 모션 감지. ROI 영역 지원."""

    def __init__(self, roi: list[int] | None = None):
        """
        Args:
            roi: 관심 영역 [x, y, width, height] 픽셀 좌표. None이면 전체 프레임 사용.
        """
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True,
        )
        self._roi = roi

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        if self._roi is None:
            return frame
        x, y, w, h = self._roi
        fh, fw = frame.shape[:2]
        # 좌표가 프레임 범위를 벗어나지 않도록 클램핑
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)
        return frame[y:y + h, x:x + w]

    def detect(self, frame: np.ndarray) -> dict:
        """
        프레임의 움직임을 분석한다.

        Returns:
            {
                "motion_score": float (0~100),
                "presence_score": float (0~100, ROI 내 전경 픽셀 비율),
                "foreground_mask": np.ndarray (디버깅/시각화용),
            }
        """
        region = self._crop_roi(frame)

        # 노이즈 제거를 위한 Gaussian blur
        blurred = cv2.GaussianBlur(region, (21, 21), 0)

        # 배경 차분 → 전경 마스크
        fg_mask = self._bg_subtractor.apply(blurred)

        # 그림자(값 127) 제거 → 이진화 (움직임 = 255)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

        # 움직임 점수: 전경 픽셀 비율 (0~100)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        if total_pixels == 0:
            return {"motion_score": 0.0, "presence_score": 0.0, "foreground_mask": thresh}

        foreground_pixels = cv2.countNonZero(thresh)
        motion_score = (foreground_pixels / total_pixels) * 100.0

        # 존재감 점수: 배경 모델 대비 현재 프레임의 차이 (아기가 침대에 있는지)
        # 배경 모델에서 크게 벗어난 픽셀의 비율
        bg_image = self._bg_subtractor.getBackgroundImage()
        presence_score = 0.0
        if bg_image is not None:
            bg_region = self._crop_roi(bg_image)
            if bg_region.shape == region.shape:
                diff = cv2.absdiff(region, bg_region)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
                _, presence_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                presence_pixels = cv2.countNonZero(presence_mask)
                presence_score = (presence_pixels / total_pixels) * 100.0

        return {
            "motion_score": round(motion_score, 2),
            "presence_score": round(presence_score, 2),
            "foreground_mask": thresh,
        }
