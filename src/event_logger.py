import csv
import logging
import os
import sqlite3

import cv2
import numpy as np

from src.state_tracker import Event

logger = logging.getLogger(__name__)


class EventLogger:
    """이벤트를 SQLite DB + CSV에 기록하고, 스냅샷 이미지를 저장."""

    def __init__(self, database_path: str, snapshot_dir: str, csv_path: str):
        self._db_path = database_path
        self._snapshot_dir = snapshot_dir
        self._csv_path = csv_path

        os.makedirs(os.path.dirname(database_path) or ".", exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

        self._conn = sqlite3.connect(database_path)
        self._init_db()
        self._init_csv()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera TEXT NOT NULL,
                event_type TEXT NOT NULL,
                confidence REAL,
                snapshot_path TEXT,
                duration_minutes REAL,
                notes TEXT
            )
        """)
        self._conn.commit()

    def _init_csv(self):
        if not os.path.exists(self._csv_path):
            os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "camera", "event_type", "confidence",
                    "snapshot_path", "duration_minutes", "notes",
                ])

    def log(self, event: Event, frame: np.ndarray | None = None) -> int:
        """이벤트를 DB/CSV에 기록하고, 프레임이 있으면 스냅샷 저장."""
        snapshot_path = ""
        if frame is not None:
            snapshot_path = self._save_snapshot(event, frame)

        # 타임스탬프를 mm:ss 형식으로 변환
        minutes = int(event.timestamp // 60)
        seconds = int(event.timestamp % 60)
        ts_str = f"{minutes:02d}:{seconds:02d}"

        # duration 계산 (end 이벤트의 경우 notes에서 추출)
        duration = None
        if "duration=" in event.notes:
            try:
                dur_str = event.notes.split("duration=")[1].split("s")[0]
                duration = round(float(dur_str) / 60.0, 1)
            except (ValueError, IndexError):
                pass

        # SQLite 저장
        cursor = self._conn.execute(
            "INSERT INTO events (timestamp, camera, event_type, confidence, snapshot_path, duration_minutes, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts_str, event.camera, event.event_type, event.confidence,
             snapshot_path, duration, event.notes),
        )
        self._conn.commit()

        # CSV 추가
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts_str, event.camera, event.event_type, event.confidence,
                snapshot_path, duration, event.notes,
            ])

        logger.info(
            "이벤트 기록: [%s] %s - %s (confidence=%.2f)",
            ts_str, event.camera, event.event_type, event.confidence,
        )
        return cursor.lastrowid

    def _save_snapshot(self, event: Event, frame: np.ndarray) -> str:
        minutes = int(event.timestamp // 60)
        seconds = int(event.timestamp % 60)
        filename = f"{event.camera}_{event.event_type}_{minutes:02d}m{seconds:02d}s.jpg"
        path = os.path.join(self._snapshot_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def get_summary(self) -> list[dict]:
        """기록된 모든 이벤트를 반환."""
        cursor = self._conn.execute(
            "SELECT timestamp, camera, event_type, confidence, duration_minutes, notes "
            "FROM events ORDER BY id"
        )
        rows = cursor.fetchall()
        return [
            {
                "timestamp": r[0], "camera": r[1], "event_type": r[2],
                "confidence": r[3], "duration_minutes": r[4], "notes": r[5],
            }
            for r in rows
        ]

    def close(self):
        self._conn.close()
