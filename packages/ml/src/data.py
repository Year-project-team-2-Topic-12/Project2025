from pathlib import Path
import pickle
from IPython.display import display
import os
import numpy as np
from skimage import io
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .env import MODELS_PATH, DATA_PATH, RESULTS_PATH, DATA_ROOT
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def get_anatomy_suffix(anatomy: str | None = None) -> str:
    if anatomy is None:
        return "_all"
    return f"_{anatomy}"

def get_results_fname(model_name_base: str, use_all: bool) -> str:
    suffix = "_all" if use_all else ""
    return f"results_{model_name_base}{suffix}.csv"

def get_model_fname(model_name_base: str, anatomy: str | None = None) -> str:
    return f"model_{model_name_base}{get_anatomy_suffix(anatomy)}.pkl"

def get_models_path(model_fname: str) -> Path:
    path = Path(MODELS_PATH) / Path(model_fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_results_path(results_fname: str) -> Path:
    path = Path(RESULTS_PATH) / Path(results_fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_data_path(data_fname: str) -> Path:
    path = Path(DATA_PATH) / Path(data_fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_pickle(filepath: Path):
    try:
        logger.debug("Загрузка данных из %s", filepath)
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error("Файл %s не найден.", filepath)
        return None

def save_pickle(data, filepath: Path):
    logger.debug("Сохранение данных в %s", filepath)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def save_model_pipeline(model, model_name_base: str, anatomy: str | None = None):
    model_fname = get_model_fname(model_name_base, anatomy)
    filepath = get_models_path(model_fname)
    logger.debug("Сохранение модели в %s", filepath)
    save_pickle(model, filepath)


def load_model_pipeline(model_name_base: str, anatomy: str | None = None):
    model_fname = get_model_fname(model_name_base, anatomy)
    filepath = get_models_path(model_fname)
    logger.debug("Загрузка модели из %s", filepath)
    return load_pickle(filepath)

def save_model_results(results: pd.DataFrame, model_name_base: str, use_all: bool) -> GridSearchCV:
    results_fname = get_results_fname(model_name_base, use_all)
    filepath = get_results_path(results_fname)
    logger.debug("Сохранение результатов в %s", filepath)
    results.to_csv(filepath, index=False)

def load_model_results(model_name_base: str, use_all: bool) -> pd.DataFrame | None:
    results_fname = get_results_fname(model_name_base, use_all=use_all)
    filepath = get_results_path(results_fname)
    try:
        logger.debug("Загрузка результатов из %s", filepath)
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error("Файл %s не найден.", filepath)
        return None

class ImageLoader(BaseEstimator, TransformerMixin):
    def __init__(self, as_gray=True):
        self.as_gray = as_gray

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X — это массив путей (список строк)
        imgs = []
        for path in X:
            logger.debug("Загрузка изображения %s", path)
            imgs.append(io.imread(path, as_gray=self.as_gray))
        return np.array(imgs)


def parse_study_path(study_path: str, return_images=False):
    parts = Path(study_path).parts
    split = parts[-4]
    anatomy = parts[-3]
    patient_id = parts[-2]
    study_name = parts[-1]
    label = 1 if 'positive' in study_name else 0
    if return_images:
        image_records = []
        images_paths = os.listdir(study_path)
        study_name=Path(study_path).name
        for f in images_paths:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_records.append({
                    'path': os.path.join(study_path, f),
                    'split': split,
                    'anatomy': anatomy,
                    'patient_id': patient_id,
                    'study_id': study_name,
                    'label': label,
                })
        return image_records
    else:
        return {
            'split': split,
            'anatomy': anatomy,
            'patient_id': patient_id,
            'study_id': study_name,
            'label': label,
            'path': study_path
        }


"""
Строит DataFrame со всеми исследованиями в структуре папок root_dir.
Если return_images=True, возвращает DataFrame с путями картинок вместо исследований.
"""
def build_dataframe(root_dir: str = DATA_ROOT, return_images=False):
    records = []
    logger.info("Загружаем датафрейм с данными из корня: %s", root_dir)
    for split in ['train', 'valid']:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            continue
        anatomies_paths = os.listdir(split_path)
        for _idx, anatomy in tqdm(enumerate(anatomies_paths), total=len(anatomies_paths),  desc=f"Обработка анатомий в {split}"):
            anatomy_path = os.path.join(split_path, anatomy)
            if not os.path.isdir(anatomy_path):
                continue
            patients_paths = os.listdir(anatomy_path)
            for patient in patients_paths:
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
