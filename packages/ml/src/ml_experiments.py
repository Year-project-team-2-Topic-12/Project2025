import numpy as np
import cv2
import pandas as pd
import os
from tqdm.auto import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from ml.hog import compute_study_hog
from pathlib import Path
import contextlib
from .env import MODELS_PATH, DATA_PATH, RESULTS_PATH, DATA_ROOT
import logging
from ml.data import parse_study_path
logger = logging.getLogger(__name__)

# функция для применения аугментации к анатомиям
def augment(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    h, w = img.shape[:2]

    # 1) Небольшой поворот
    angle = np.random.uniform(-4, 4)
    scale = np.random.uniform(0.98, 1.02)

    # 2) Небольшой сдвиг
    tx = np.random.uniform(-0.02, 0.02) * w
    ty = np.random.uniform(-0.02, 0.02) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    img = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # 3) Небольшое изменение яркости/контраста
    alpha = np.random.uniform(0.9, 1.1)   # contrast
    beta = np.random.uniform(-0.04, 0.04) # brightness
    img = np.clip(img * alpha + beta, 0, 1)

    # 4) Иногда очень слабый blur
    if np.random.rand() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=np.random.uniform(0.1, 0.5))

    return np.clip(img, 0, 1)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return img.astype(np.float32) / 255.0

def save_image(img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok = cv2.imwrite(out_path, img_uint8)
    if not ok:
        raise ValueError(f"Не удалось сохранить изображение: {out_path}")

def augmented_df(images_df: pd.DataFrame, out_root: str, anatomies):
    parts = []

    for anatomy in anatomies:
        df_anatomy = images_df[images_df["anatomy"] == anatomy].copy()
        if len(df_anatomy) == 0:
            continue

        saved_rows = []

        for idx, row in tqdm(
            df_anatomy.iterrows(),
            total=len(df_anatomy),
            desc=f"Сохранение {anatomy}"
        ):
            src_path = row["path"]
            img = load_image(src_path)

            patient_id = row["patient_id"]
            study_id = row["study_id"]
            split = row["split"]

            filename = os.path.basename(src_path)
            base_name, ext = os.path.splitext(filename)

            original_out_dir = os.path.join(
                out_root, "original", split, anatomy, patient_id, study_id
            )
            original_out_path = os.path.join(original_out_dir, filename)
            save_image(img, original_out_path)

            if idx < 2:
                print("ORIG:", original_out_path)

            orig_row = row.copy()
            orig_row["path"] = original_out_path
            orig_row["source"] = "original_saved"
            saved_rows.append(orig_row)

            if split == "train":
                img_aug = augment(img)

                aug_out_dir = os.path.join(
                    out_root, "augmented", split, anatomy, patient_id, study_id
                )
                aug_out_path = os.path.join(aug_out_dir, f"{base_name}_aug_{idx}{ext}")
                save_image(img_aug, aug_out_path)

                if idx < 2:
                    print("AUG :", aug_out_path)

                aug_row = row.copy()
                aug_row["path"] = aug_out_path
                aug_row["source"] = "augmented"
                saved_rows.append(aug_row)

        parts.append(pd.DataFrame(saved_rows, columns=list(df_anatomy.columns) + ["source"]))

    return pd.concat(parts, ignore_index=True)

# определение параметров в зависимости от текущей яркости изображений
def parameters(mean, std):
    # Контраст
    if std < 30:
        alpha = 1.3
    elif std < 50:
        alpha = 1.1
    else:
        alpha = 1.0

    # Яркость
    if mean < 40:
        beta = 20
    elif mean < 70:
        beta = 10
    else:
        beta = 0

    clip_limit = 3

    return alpha, beta, clip_limit

# функция сохранения изображений
def save_processed(img, original_path, base_input_dir, base_output_dir):
    rel_path = os.path.relpath(original_path, base_input_dir)
    output_path = os.path.join(base_output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

# обработка и сохранение всех изображений
def process_images(df_subset, base_input_dir, base_output_dir, tile_grid_size=(10, 10)):
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc='Обработка изображений'):
        path = row['path']
        mean = row['mean']
        std = row['std']

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        alpha, beta, clip_limit = parameters(mean, std)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img)
        img_final = cv2.convertScaleAbs(img_clahe, alpha=alpha, beta=beta)

        save_processed(img_final, path, base_input_dir, base_output_dir)

#сбор датафрейма из EDA

def get_brightness_stats(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Переводим RGB в grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stats = {
        'min': int(img.min()),
        'max': int(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std()),
        'range': int(img.max() - img.min())
    }
    return stats

def collect_image_dimensions(root_dir: Path, extensions):
    records = []
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in extensions and not path.name.startswith("._"):
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception as e:
                print(f"⚠️ Ошибка при чтении {path}: {e}")
                continue
            records.append(
                {
                    "path": path,
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h if h != 0 else np.nan,
                }
            )
    df = pd.DataFrame(records)
    print(f"Всего найдено изображений: {len(df)}")
    return df


def resize_with_padding(img: Image.Image,
                        target_h,
                        target_w) -> Image.Image:
    """
    Ресайз изображения с сохранением пропорций + центрированный паддинг
    до (target_h, target_w).
    """
    w, h = img.size

    # Первый шаг: пробуем масштабировать по высоте
    if h > 0:
        scale = target_h / h
    else:
        scale = 1.0

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Если по высоте всё хорошо, но ширина вдруг слишком большая — масштабируем по ширине
    if new_w > target_w:
        scale = target_w / w
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

    # Собственно ресайз
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Паддинг до нужного размера
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # Фон чёрный (0), под рентгены как раз
    img_padded = ImageOps.expand(
        img_resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0
    )

    return img_padded


def save_resized_dataset(df: pd.DataFrame,
                         input_root: Path,
                         output_root: Path,
                         target_h,
                         target_w
                         ):
    num_saved = 0
    for i, row in tqdm(df.iterrows(), total=len(df),desc="Ресайз изображений..."):
        src_path: Path = row["path"]
        rel_path = src_path.relative_to(input_root)  # относительный путь от корня
        dst_path = output_root / rel_path

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src_path) as img:
                img = img.convert("L")  # на всякий случай делаем grayscale
                img_resized = resize_with_padding(img, target_h, target_w)
                img_resized.save(dst_path)
            num_saved += 1
        except Exception as e:
            print(f"⚠️ Ошибка при обработке {src_path}: {e}")

    print(f"Готово! Сохранено изображений: {num_saved}")



def hog_data(images_df: pd.DataFrame, anatomy: str, is_images=False):
    images_subset = images_df[images_df["anatomy"] == anatomy]
    X_all, y_all, study_ids, splits, anatomies = compute_study_hog(
      images_subset,
      is_images=is_images
      )

    return {
        "X": X_all,
        "y": y_all,
        "study_ids": study_ids,
        "splits": splits,
        "anatomy": anatomies
    }

def hog_dataset(images_df: pd.DataFrame, anatomies):
    all_rows = []

    for anatomy in anatomies:
        print(f'Обработка {anatomy}')
        df_part = images_df[images_df["anatomy"] == anatomy].copy()
        data = hog_data(df_part, anatomy, is_images=True)

        X = data["X"]
        y = data["y"]
        splits = data["splits"]
        study_ids = data["study_ids"]
        anatomy_list = data["anatomy"]

        for i in range(len(y)):
            all_rows.append({
                "anatomy": anatomy_list[i],
                "study_id": study_ids[i],
                "split": splits[i],
                "label": y[i],
                "hog": X[i]
            })
    return pd.DataFrame(all_rows)

def split_hog_features(hog_df: pd.DataFrame, anatomy: str):
    df_part = hog_df[hog_df["anatomy"] == anatomy].copy()

    X = np.vstack(df_part["hog"].values).astype(np.float32)
    y = df_part["label"].values
    splits = df_part["split"].values

    X_train = X[splits == "train"]
    y_train = y[splits == "train"]
    X_val = X[splits == "valid"]
    y_val = y[splits == "valid"]

    return X_train, y_train, X_val, y_val

def build_exp_dataframe(root_dir: str = DATA_ROOT, return_images=False):
    records = []
    logger.info("Загружаем датафрейм с данными из корня: %s", root_dir)
    for split in ['train', 'valid']:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            continue
        anatomies_paths = os.listdir(split_path)
        for _idx, anatomy in tqdm(
              enumerate(anatomies_paths),
              total=len(anatomies_paths),
              desc=f"Загрузка анатомий из {split}"):

            anatomy_path = os.path.join(split_path, anatomy)

            patients_paths = os.listdir(anatomy_path)
            print(f'Загрузка {anatomy}')
            for patient in tqdm(patients_paths, desc="Исследования..", leave=False):
                patient_path = os.path.join(anatomy_path, patient)
                study_paths = os.listdir(patient_path)
                for study in study_paths:
                    study_path = os.path.join(patient_path, study)
                    if os.path.isdir(study_path):
                        records_to_add = parse_study_path(study_path=study_path, return_images=return_images)
                        if return_images:
                            records.extend(records_to_add)
                        else:
                            records.append(parse_study_path(study_path))
    return pd.DataFrame(records)

