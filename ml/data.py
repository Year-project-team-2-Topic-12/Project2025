from pathlib import Path
import pickle
from IPython.display import display
import os
import numpy as np
from skimage import io
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def save_model_data(model, name):
    filepath = Path('models') / Path(name)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model_data(name):
    try:
        filepath = Path('models') / Path(name)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File {name} not found.")
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


