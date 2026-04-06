import argparse
import logging
import sys

import yaml

from src.camera import VideoCapture
from src.motion_detector import MotionDetector
from src.activity_classifier import ActivityClassifier
from src.state_tracker import BedroomTracker, LivingRoomTracker
from src.frame_sampler import FrameSampler
from src.event_logger import EventLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_bedroom(cam_config: dict, event_logger: EventLogger):
    """침실 영상 처리: 모션 감지 기반 수면/기상 감지 (100% 로컬)."""
    source = cam_config["source"]
    logger.info("=== 침실 영상 처리 시작: %s ===", source)

    roi = cam_config.get("roi")
    motion_threshold = cam_config.get("motion_threshold", 5)
    presence_minutes = cam_config.get("presence_threshold", 3)
    frame_interval = cam_config.get("frame_interval_seconds", 2)

    detector = MotionDetector(roi=roi)
    tracker = BedroomTracker(
        motion_threshold=motion_threshold,
        presence_threshold_minutes=presence_minutes,
    )

    with VideoCapture(source, frame_interval) as cap:
        if cap.fps == 0:
            logger.error("영상을 열 수 없습니다: %s", source)
            return

        frame_count = 0
        while True:
            frame, timestamp = cap.read_frame()
            if frame is None:
                break

            result = detector.detect(frame)
            event = tracker.update(result["motion_score"], result["presence_score"], timestamp)

            if event:
                event_logger.log(event, frame)

            frame_count += 1
            if frame_count % 100 == 0:
                progress = cap.progress * 100
                logger.info(
                    "침실 진행률: %.1f%% | 상태: %s | 모션: %.1f | 존재감: %.1f",
                    progress, tracker.state.value,
                    result["motion_score"], result["presence_score"],
                )

    logger.info("=== 침실 영상 처리 완료 (%d 프레임) ===", frame_count)


def process_livingroom(cam_config: dict, gemini_config: dict, event_logger: EventLogger):
    """거실 영상 처리: 모션 감지 + Gemini Flash 활동 분류."""
    source = cam_config["source"]
    logger.info("=== 거실 영상 처리 시작: %s ===", source)

    motion_threshold = cam_config.get("motion_threshold", 15)
    frame_interval = cam_config.get("frame_interval_seconds", 2)
    api_interval = cam_config.get("api_interval_seconds", 30)

    detector = MotionDetector()
    tracker = LivingRoomTracker()
    sampler = FrameSampler(
        motion_threshold=motion_threshold,
        api_interval_seconds=api_interval,
    )

    # Gemini API 초기화
    try:
        classifier = ActivityClassifier(
            model=gemini_config.get("model", "gemini-2.0-flash"),
            api_key_env=gemini_config.get("api_key_env", "GEMINI_API_KEY"),
        )
    except ValueError as e:
        logger.error("Gemini API 초기화 실패: %s", e)
        logger.info("거실 영상 처리를 건너뜁니다.")
        return

    with VideoCapture(source, frame_interval) as cap:
        if cap.fps == 0:
            logger.error("영상을 열 수 없습니다: %s", source)
            return

        frame_count = 0
        while True:
            frame, timestamp = cap.read_frame()
            if frame is None:
                break

            result = detector.detect(frame)
            motion_score = result["motion_score"]

            if sampler.should_call_api(motion_score, timestamp):
                classification = classifier.classify(frame)
                sampler.mark_api_called(timestamp)

                event = tracker.update(classification, timestamp)
                if event:
                    event_logger.log(event, frame)

                logger.debug(
                    "[%.0fs] API 호출 #%d: %s (confidence=%.2f)",
                    timestamp, classifier.call_count,
                    classification.get("activity"), classification.get("confidence", 0),
                )

            frame_count += 1
            if frame_count % 100 == 0:
                progress = cap.progress * 100
                logger.info(
                    "거실 진행률: %.1f%% | 상태: %s | 모션: %.1f | API 호출: %d회",
                    progress, tracker.state.value, motion_score, classifier.call_count,
                )

    logger.info(
        "=== 거실 영상 처리 완료 (%d 프레임, API %d회 호출) ===",
        frame_count, classifier.call_count,
    )


def print_summary(event_logger: EventLogger):
    """감지된 이벤트 요약 출력."""
    events = event_logger.get_summary()
    if not events:
        print("\n감지된 이벤트가 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  감지된 이벤트 요약 ({len(events)}건)")
    print(f"{'='*60}")
    for ev in events:
        duration_str = ""
        if ev["duration_minutes"]:
            duration_str = f" ({ev['duration_minutes']:.1f}분)"
        print(
            f"  [{ev['timestamp']}] {ev['camera']:>10s} | "
            f"{ev['event_type']:<22s} | "
            f"신뢰도 {ev['confidence']:.0%}{duration_str}"
        )
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="BabyCV - 아기 활동 감지 시스템")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument(
        "--camera", choices=["bedroom", "livingroom", "all"], default="all",
        help="처리할 카메라 (기본: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="디버그 로그 출력")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    log_config = config.get("logging", {})
    event_logger = EventLogger(
        database_path=log_config.get("database_path", "data/events.db"),
        snapshot_dir=log_config.get("snapshot_dir", "data/snapshots"),
        csv_path=log_config.get("csv_path", "data/events.csv"),
    )

    try:
        cameras = config.get("cameras", {})

        if args.camera in ("bedroom", "all") and "bedroom" in cameras:
            process_bedroom(cameras["bedroom"], event_logger)

        if args.camera in ("livingroom", "all") and "livingroom" in cameras:
            gemini_config = config.get("gemini", {})
            process_livingroom(cameras["livingroom"], gemini_config, event_logger)

        print_summary(event_logger)
    finally:
        event_logger.close()


if __name__ == "__main__":
    main()
