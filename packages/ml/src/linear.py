from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

def extract_intensity_features_from_study(study_path):
    study_dir = Path(study_path)
    image_files = list(study_dir.glob("*.png")) + list(study_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"Нет изображений в: {study_dir}")
    img_path = image_files[0]
    img = Image.open(img_path).convert('L')
    flat = np.array(img).ravel().astype(np.float32) / 255.0
    return {
        'mean': np.mean(flat),
        'std': np.std(flat),
        'median': np.median(flat),
        'skewness': skew(flat),
        'kurtosis': kurtosis(flat),
        'entropy': entropy(np.histogram(flat, bins=64, density=True)[0] + 1e-8),
        'p90': np.percentile(flat, 90),
        'p10': np.percentile(flat, 10)
    }

def prepare_data_for_anatomy(paths_df: pd.DataFrame, anatomy: str | None = None, get_all=False, extract_func=extract_intensity_features_from_study):
    train_df = paths_df[paths_df['split'] == 'train']
    valid_df = paths_df[paths_df['split'] == 'valid']
    if not get_all:
        train_df = train_df[train_df['anatomy'] == anatomy]
        valid_df = valid_df[valid_df['anatomy'] == anatomy]
    # Извлечение признаков
    X_train_list, y_train_list = [], []
    for _, row in train_df.iterrows():
        try:
            feats = extract_func(row['path'])
            X_train_list.append(feats)
            y_train_list.append(row['label'])
        except Exception:
            continue

    X_val_list, y_val_list = [], []
    for _, row in valid_df.iterrows():
        try:
            feats = extract_func(row['path'])
            X_val_list.append(feats)
            y_val_list.append(row['label'])
        except Exception:
            continue

    if len(X_train_list) == 0 or len(X_val_list) == 0:
        return None, None, None, None

    # Преобразуем в датафреймы/массивы
    X_train = pd.DataFrame(X_train_list)
    y_train = np.array(y_train_list, dtype=int)
    X_val = pd.DataFrame(X_val_list)
    y_val = np.array(y_val_list, dtype=int)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    return X_train, y_train, X_val, y_val


