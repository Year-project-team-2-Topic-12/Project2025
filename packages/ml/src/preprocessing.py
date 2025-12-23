import cv2
import numpy as np
from .env import IMAGE_SIZE

from typing import Tuple


def resize_with_padding_cv2(
    img: np.ndarray,
    *, # после этого - только кейворд-аргументы
    target_h: int = IMAGE_SIZE,
    target_w: int = IMAGE_SIZE,
    pad_value: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Ресайз с сохранением пропорций + центрированный паддинг до (target_h, target_w).
    Вход: img grayscale (H,W) или BGR/RGB (H,W,3)
    Выход: изображение того же числа каналов, dtype сохранится (обычно uint8)
    """
    if img is None:
        raise ValueError("img is None")

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image shape: {img.shape}")

    # Масштаб как в ноутбуке: сначала по высоте, если ширина не влезает — по ширине
    if h > 0:
        scale = target_h / h
    else:
        scale = 1.0

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if new_w > target_w:
        scale = target_w / w
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Паддинг до целевого размера
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return padded


def enhance_brightness_cv2(
    img_gray: np.ndarray,
    *,
    tile_grid_size: Tuple[int, int] = (10, 10),
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """
    Усиление контраста/яркости как в ноутбуке:
    CLAHE (clipLimit зависит от контраста) -> convertScaleAbs(alpha, beta)

    Вход: grayscale uint8 (H,W)
    Выход: grayscale uint8 (H,W)
    """
    if img_gray is None:
        raise ValueError("img_gray is None")
    if img_gray.ndim != 2:
        raise ValueError(f"Expected grayscale (H,W), got shape={img_gray.shape}")

    if img_gray.dtype != np.uint8:
        raise ValueError(f"Expected uint8 grayscale image, got dtype={img_gray.dtype}")

    # Если mean/std не дали — считаем по картинке (как в твоём brightness_df)
    if mean is None:
        mean = float(img_gray.mean())
    if std is None:
        std = float(img_gray.std())

    # ---- Логика параметров как в ноутбуке ----
    # Контраст (alpha)
    if std < 30:
        alpha = 1.3
    elif std < 50:
        alpha = 1.1
    else:
        alpha = 1.0

    # Яркость (beta)
    if mean < 40:
        beta = 20
    elif mean < 70:
        beta = 10
    else:
        beta = 0

    # CLAHE clipLimit фиксирован как в ноутбуке
    clip_limit = 3.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_gray)
    img_final = cv2.convertScaleAbs(img_clahe, alpha=alpha, beta=beta)
    return img_final
