import numpy as np
from PIL import Image


def pil2cv(img: Image) -> np.ndarray:
    return np.asarray(img)


def cv2pil(img: np.ndarray) -> Image:
    return Image.fromarray(img)