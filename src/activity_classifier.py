import base64
import json
import logging
import os
import time

import cv2
import numpy as np
from google import genai

logger = logging.getLogger(__name__)

LIVINGROOM_PROMPT = """이 이미지는 가정의 거실/주방 홈캠으로 촬영된 장면입니다.
1세 아기가 보인다면, 현재 활동을 분석해주세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{"baby_visible": true/false, "activity": "eating"|"diaper_change"|"playing"|"other"|"none", "confidence": 0.0-1.0, "details": "간단한 설명"}

활동 판단 기준:
- eating: 아기가 식탁/유아 의자에 앉아 음식을 먹거나, 부모가 먹여주는 장면
- diaper_change: 아기의 기저귀를 교체하는 장면 (아기가 누워있고 부모가 기저귀 처리 중)
- playing: 아기가 놀고 있는 장면
- other: 위 카테고리에 해당하지 않는 기타 활동
- none: 아기가 보이지 않거나, 특별한 활동이 없는 경우"""


class ActivityClassifier:
    """Gemini Flash API를 사용한 거실/주방 활동 분류."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key_env: str = "GEMINI_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"환경변수 {api_key_env}가 설정되지 않았습니다. "
                f"export {api_key_env}=your_api_key 로 설정하세요."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._call_count = 0

    def classify(self, frame: np.ndarray) -> dict:
        """
        프레임을 Gemini Flash로 분류한다.

        Returns:
            {"baby_visible": bool, "activity": str, "confidence": float, "details": str}
            에러 시 {"baby_visible": False, "activity": "none", "confidence": 0.0, "details": "error: ..."}
        """
        default = {"baby_visible": False, "activity": "none", "confidence": 0.0, "details": ""}

        # 프레임을 640x480 JPEG로 인코딩
        resized = cv2.resize(frame, (640, 480))
        _, jpeg_buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = jpeg_buf.tobytes()

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    genai.types.Content(
                        parts=[
                            genai.types.Part.from_text(text=LIVINGROOM_PROMPT),
                            genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        ]
                    )
                ],
            )
            self._call_count += 1

            text = response.text.strip()
            # JSON 블록이 ```json ... ``` 으로 감싸져 있을 수 있음
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(text)
            # 필수 필드 검증
            result.setdefault("baby_visible", False)
            result.setdefault("activity", "none")
            result.setdefault("confidence", 0.0)
            result.setdefault("details", "")
            return result

        except json.JSONDecodeError as e:
            logger.warning("Gemini 응답 JSON 파싱 실패: %s", e)
            default["details"] = f"json_error: {e}"
            return default
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning("Gemini rate limit 도달. 60초 대기.")
                time.sleep(60)
                default["details"] = "rate_limited"
            else:
                logger.error("Gemini API 호출 실패: %s", e)
                default["details"] = f"api_error: {e}"
            return default

    @property
    def call_count(self) -> int:
        return self._call_count
