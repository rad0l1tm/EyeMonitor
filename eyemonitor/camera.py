"""One capture/inference worker, one latest result, no unbounded frame queue."""
import threading
import time
import traceback
from dataclasses import dataclass


@dataclass
class Sample:
    sequence: int = 0
    time: float = 0.
    features: object = None
    error: str = ''
    reason: str = ''
    fps: float = 0.


class Camera(threading.Thread):
    def __init__(self, device, model, fps=15, width=640, height=480):
        super().__init__(daemon=True)
        self.device, self.model = device, model
        self.fps, self.width, self.height = fps, width, height
        self.stop_event = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()
        self.lock = threading.Lock()
        self.latest = Sample()

    def snapshot(self):
        with self.lock:
            return self.latest

    def publish(self, sample):
        with self.lock:
            self.latest = sample

    def run(self):
        cap = None
        try:
            import cv2
            import mediapipe as mp
            from . import core
            from .core import eye_features
            cv2.setNumThreads(1)
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f'Не удалось открыть камеру {self.device}; проверьте /dev/video*')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model),
                                                  delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1, min_face_detection_confidence=.6,
                min_face_presence_confidence=.6, min_tracking_confidence=.6,
                output_facial_transformation_matrixes=False)
            sequence, stamp, failures = 0, 0, 0
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                while not self.stop_event.is_set():
                    start = time.monotonic()
                    ok, frame = cap.read()
                    if not ok:
                        failures += 1
                        self.publish(Sample(sequence, start, error='Нет кадра с камеры'))
                        if failures >= 30:
                            raise RuntimeError('Камера отключилась; перезапустите приложение')
                        self.stop_event.wait(.1)
                        continue
                    failures = 0
                    if not self.enabled.is_set():
                        self.stop_event.wait(.1)
                        continue
                    if frame.shape[1] > self.width:
                        h = round(frame.shape[0] * self.width / frame.shape[1])
                        frame = cv2.resize(frame, (self.width, h))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stamp = max(stamp + 1, int(start * 1000))
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
                    features, reason = None, 'MediaPipe не нашёл лицо'
                    if result.face_landmarks:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        features = eye_features(gray, result.face_landmarks[0],
                                             gray.shape[1], gray.shape[0])
                        reason = '' if features is not None else 'lol'
                    sequence += 1
                    elapsed = time.monotonic() - start
                    self.publish(Sample(sequence, time.monotonic(), features, reason=reason,
                                        fps=1 / max(elapsed, 1 / self.fps)))
                    self.stop_event.wait(max(0, 1 / self.fps - elapsed))
        except Exception as e:
            traceback.print_exc()
            self.publish(Sample(error=f'{type(e).__name__}: {e}'))
        finally:
            if cap is not None:
                cap.release()
