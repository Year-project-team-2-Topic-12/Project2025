from pathlib import Path
import pickle
from IPython.display import display
import os
import numpy as np
from skimage import io
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import GridSearchCV


PATH_MODELS = Path('models')
PATH_MODELS.mkdir(parents=True, exist_ok=True)

def get_anatomy_suffix(anatomy: str | None = None) -> str:
    if anatomy is None:
        return "_all"
    return f"_{anatomy}"

def get_results_fname(model_name_base: str) -> str:
    return f"results_{model_name_base}.csv"

def get_model_fname(model_name_base: str, anatomy: str | None = None) -> str:
    return f"model_{model_name_base}{get_anatomy_suffix(anatomy)}.pkl"

def get_models_path(model_fname: str) -> Path:
    path = PATH_MODELS / Path(model_fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_model_data(model, model_name_base: str, anatomy: str | None = None):
    model_fname = get_model_fname(model_name_base, anatomy)
    filepath = get_models_path(model_fname)
    print(f"Сохранение модели в {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model_data(model_name_base: str, anatomy: str | None = None):
    try:
        model_fname = get_model_fname(model_name_base, anatomy)
        filepath = get_models_path(model_fname)
        with open(filepath, "rb") as f:
            print(f"Загрузка модели из {filepath}")
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Файл {model_fname} не найден.")
        return None
    
def save_model_results(results: pd.DataFrame, model_name_base: str) -> GridSearchCV:
    results_fname = get_results_fname(model_name_base)
    filepath = get_models_path(results_fname)
    print(f"Сохранение результатов в {filepath}")
    results.to_csv(filepath, index=False)

def load_model_results(model_name_base: str) -> pd.DataFrame | None:
    results_fname = get_results_fname(model_name_base)
    filepath = get_models_path(results_fname)
    try:
        print(f"Загрузка результатов из {filepath}")
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Файл {results_fname} не найден.")
        return None

class ImageLoader(BaseEstimator, TransformerMixin):
    def __init__(self, as_gray=True):
        self.as_gray = as_gray

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X — это массив путей (список строк)
        imgs = [
            print(path) or io.imread(path, as_gray=self.as_gray)
            for path in X
        ]
        return np.array(imgs)


def parse_study_path(study_path: str):
    parts = Path(study_path).parts
    split = parts[-4]
    anatomy = parts[-3]
    patient_id = parts[-2]
    study_name = parts[-1]
    label = 1 if 'positive' in study_name else 0
    return {
        'split': split,
        'anatomy': anatomy,
        'patient_id': patient_id,
        'study_id': study_name,
        'label': label,
        'path': study_path
    }

def build_dataframe(root_dir: str):
    records = []
    for split in ['train', 'valid']:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            continue
        for anatomy in os.listdir(split_path):
            anatomy_path = os.path.join(split_path, anatomy)
            if not os.path.isdir(anatomy_path):
                continue
            for patient in os.listdir(anatomy_path):
                patient_path = os.path.join(anatomy_path, patient)
                for study in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study)
                    if os.path.isdir(study_path):
                        records.append(parse_study_path(study_path))
    return pd.DataFrame(records)
