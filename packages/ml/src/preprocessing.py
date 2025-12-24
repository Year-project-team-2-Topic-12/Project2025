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

    # Масштаб так, чтобы ВПИСАТЬ в target (без обрезки)
    scale = min(target_w / w, target_h / h)
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
    tile_grid_size: Tuple[int, int] = (8, 8),
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
        # CLAHE в OpenCV ожидает uint8
        img_u8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img_gray

    # Если mean/std не дали — считаем по картинке (как в твоём brightness_df)
    if mean is None:
        mean = float(img_u8.mean())
    if std is None:
        std = float(img_u8.std())

    # ---- Логика параметров (по смыслу твоего ноутбука) ----
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

    # CLAHE clipLimit (усиление локального контраста)
    if std < 30:
        clip_limit = 3.0
    elif std < 50:
        clip_limit = 2.0
    else:
        clip_limit = 1.5

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_u8)
    img_final = cv2.convertScaleAbs(img_clahe, alpha=alpha, beta=beta)
    return img_final
