import cv2
import json
import numpy as np
import time
import threading
from fer import FER
from fast api import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from core.auth import get_current_user

router = APIRouter()

class EmotionTracker:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.timeline = []
        self.is_running = False
        self.second = 0
        self._thread = None
        self.cap = None

    def start_tracking(self):
        """Start camera and begin sampling emotions every second"""
        self.cap = cv2.VideoCapture(0)
        self.timeline = []
        self.second = 0
        self.is_running = True
        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.start()

    def _sample_loop(self):
        """Runs in background thread - samples every 1 second"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
            try:
                result = self.detector.detect(frame)
                if result:
                    emotions = result[0]['emotions']
                    dominant = max(motions, key=emotions.get)
                    self.timeline.append({
                        "second": self.second,
                        "emotions": dominant,
                        "confidence": round(emotions[dominant], 2),
                        "all": {k: round(v, 2) for k, v in emotions.items()}
                    })
            except Exception as e:
                pass

            self.second += 1
            time.sleep(1)

    def stop_tracking(self, question: str = "", part: int = 1) -> dict:
        """Stop tracking and return emotion JSON"""
        self.is_running = False
        if self._thread:
            self._thread.join()
        if self.cap:
            self.cap.release()

        return self._build_result(question, part)

    def _build_result(self, question: str, part: int) -> dict:
        timeline = self.timeline
        total = len(timeline) or 1

        emotion_counts = {}
        for entry in timeline:
            em = entry['emotion']
            emotion_counts[em] = emotion_counts.get(em, 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        nervous_pct = ((emotion_counts.get('fear', 0) + emotion_counts.get('sad', 0)) / total) * 100
        happy_pct = (emotion_counts.get('happy', 0) / total) * 100
        neutral_pct = (emotion_counts.get('neutral', 0) / total) * 100
        conf_score = min(100, max(0, round(happy_pct * 0.8 + neutral_pct * 0.5 - nervous_pct * 0.5 + 50)))

        return {
            "part": part,
            "question": question,
            "duration_seconds": total,
            "emotion_timeline": timeline,
            "summary": {
                "dominant_emotion": dominant,
                "confidence_score": conf_score,
                "nervous_percentage": round(nervous_pct),
                "happy_percentage": round(happy_pct),
                "emotion_breakdown": emotion_counts
            }
        }

tracker = EmotionTracker()

@router.post("/emotion/start")
async def start_emotion_tracking(current_user=Depends(get_current_user)):
    """Call this when user starts speaking"""
    tracker.start_tracking()
    return {"status": "tracking started"}

@router.post("/emotion/stop")
async def stop_emotion_tracking(
    question: str = "",
    part: int = 1,
    current_user=Depends(get_current_user)
):
    """Call this when user submits answer — returns emotion JSON"""
    result = tracker.stop_tracking(question=question, part=part)
    return JSONResponse(content=result)
