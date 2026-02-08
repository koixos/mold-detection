from .defs import PreprocessParams, DetectParams, PreprocessedImage

from dataclasses import dataclass, field
import numpy as np
from typing import  List


@dataclass
class ImageState:
    path: str
    filename: str

    original: np.ndarray
    preprocessed: PreprocessedImage = field(default_factory=PreprocessedImage)
    detected: np.ndarray | None = None

    preprocess_params: PreprocessParams = field(default_factory=PreprocessParams)
    detect_params: DetectParams = field(default_factory=DetectParams)
    custom = False
    info = ""


class AppState:
    def __init__(self):
        self.images: List[ImageState] = []
        self.active_index: int = -1
        self._listeners = []


    def add_listener(self, callback):
        self._listeners.append(callback)


    def add_image(self, image: ImageState):
        self.images.append(image)
        self.active_index = len(self.images) - 1
        self._notify()


    def remove_image(self, index: int):
        if 0 <= index < len(self.images):
            self.images.pop(index)
            if not self.images:
                self.active_index = -1
            else:
                self.active_index = max(0, self.active_index - 1)
            self._notify()


    def clear_images(self):
        self.images.clear()
        self.active_index = -1
        self._notify()


    def set_active(self, index: int):
        if 0 <= index < len(self.images):
            self.active_index = index
            self._notify()


    def active(self) -> ImageState | None:
        if self.active_index == -1:
            return None
        return self.images[self.active_index]
    

    def _notify(self):
        for callback in self._listeners:
            callback()
    