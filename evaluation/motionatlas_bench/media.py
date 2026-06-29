from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class FrameReader:
    """Read frames from either an mp4 file or a directory of image frames."""

    def __init__(self, media_path: Path):
        self.media_path = Path(media_path)
        self.capture: Optional[cv2.VideoCapture] = None
        self.image_paths: list[Path] = []
        self.total_frames = 0
        self.fps: Optional[float] = None
        self.is_video = self.media_path.is_file()

        if self.media_path.is_dir():
            self.image_paths = sorted(
                [
                    path
                    for path in self.media_path.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda path: path.name,
            )
            self.total_frames = len(self.image_paths)
            self.fps = None
            return

        if self.media_path.is_file():
            self.capture = cv2.VideoCapture(str(self.media_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"failed to open video: {self.media_path}")
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self.fps = fps if fps > 0 else None
            return

        raise FileNotFoundError(f"media path does not exist: {self.media_path}")

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self.total_frames:
            return None
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.capture.read()
            return frame if ok else None
        image = cv2.imread(str(self.image_paths[frame_index]))
        return image

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def uniform_sample_indices(total_frames: int, num_frames: int) -> list[int]:
    total_frames = int(total_frames)
    num_frames = int(num_frames)
    if total_frames <= 0 or num_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    if num_frames == 1:
        return [0]
    return [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
