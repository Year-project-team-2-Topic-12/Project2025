from __future__ import annotations
import numpy as np
from skimage import io
from skimage.feature import hog
import pandas as pd
from .data import get_data_path
from tqdm import tqdm
from typing import Sequence, Any
import glob
import os

from sklearn.base import BaseEstimator, TransformerMixin

def get_hog_anatomy_filename(anatomy, is_images=False):
    suffix = "_images" if is_images else ""
    return f"hog_{anatomy}{suffix}.npz"

def compute_hog(img: np.ndarray, visualize=False) -> np.ndarray:
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=visualize,
    )

def compute_hog_with_visualization(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img.ndim == 2:
        img = np.array([img])
    features_list = []
    hog_images = []
    for im in img:
        features, hog_image = compute_hog(im, visualize=True)
        features_list.append(features)
        hog_images.append(hog_image)
    features = np.mean(features_list, axis=0)
    hog_image = hog_images[0]
    return features, hog_image

def compute_images_hog(
    images: Sequence[np.ndarray],
) -> np.ndarray:
    hog_vecs: list[np.ndarray] = []

    for file in images:
        feat = compute_hog(file)
        hog_vecs.append(feat)

    if not hog_vecs:
        raise ValueError("No images to compute HOG from")

    study_feat = np.mean(hog_vecs, axis=0)
    return study_feat

def compute_study_hog(df_subset, as_gray=True, is_images=False):
        print('updated 1')
        X_feats = []
        y_labels = []
        study_ids = []
        splits = []
        anatomies = []

        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Вычисление HOG"):
            study_dir = row["path"]
            if not is_images:
                img_paths = glob.glob(os.path.join(study_dir, "*.png"))
            else:
                img_paths = [row["path"]]
            hog_vecs = []
            for p in img_paths:
                img = io.imread(p, as_gray=as_gray)
                feat = compute_hog(img)
                hog_vecs.append(feat)

            if not hog_vecs:
                continue

            study_feat = np.mean(hog_vecs, axis=0)
            X_feats.append(study_feat)
            splits.append(row["split"])
            y_labels.append(row["label"])
            study_ids.append(row["study_id"])
            anatomies.append(row["anatomy"])

        X = np.vstack(X_feats)
        y = np.array(y_labels)
        return X, y, study_ids, splits, anatomies


def create_andor_return_hog_data(images_df: pd.DataFrame, anatomy: str, is_images=False) -> np.ndarray:
    print("\nОбработка анатомии:", anatomy)
    hog_filepath = get_data_path(get_hog_anatomy_filename(anatomy, is_images=is_images))
    try:
        data = np.load(hog_filepath, allow_pickle=True)
    except FileNotFoundError:
        images_subset = images_df[images_df['anatomy'] == anatomy] 
        X_all, y_all, study_ids, splits, anatomies = compute_study_hog(images_subset, is_images=is_images)
        np.savez(hog_filepath, X=X_all, y=y_all, study_ids=study_ids, splits=splits, anatomy=anatomies)
    data = np.load(hog_filepath, allow_pickle=True)
    return data

class StudyHOGTransformer(BaseEstimator, TransformerMixin):
    # ================================
    # CLASS-LEVEL SHARED CACHES (static)
    # ================================
    IMAGE_CACHE = {}   # path -> np.ndarray(img)
    STUDY_CACHE = {}   # study_dir -> np.ndarray(hog_vector)

    def __init__(self, as_gray=True, cache_images=True, cache_study=True, hog_params={}):
        self.as_gray = as_gray
        self.cache_images = cache_images
        self.cache_study = cache_study
        self.hog_params = hog_params

    # --------------------------------
    def fit(self, X, y=None):
        # Кэш не зависит от инстанса
        return self

    @classmethod
    def clear_caches(cls):
        cls.IMAGE_CACHE.clear()
        cls.STUDY_CACHE.clear()

    # --------------------------------
    def _load_image(self, path):
        # Проверяем кэш картинок
        if self.cache_images and path in StudyHOGTransformer.IMAGE_CACHE:
            return StudyHOGTransformer.IMAGE_CACHE[path]

        # Читаем с диска
        img = io.imread(path, as_gray=self.as_gray)

        # Кэшируем
        if self.cache_images:
            StudyHOGTransformer.IMAGE_CACHE[path] = img

        return img

    # --------------------------------
    def _hog_for_image(self, img):
        return hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            **self.hog_params
        )

    # --------------------------------
    def _hog_for_study(self, study_dir):
        # Проверяем кэш study
        if self.cache_study and study_dir in StudyHOGTransformer.STUDY_CACHE:
            return StudyHOGTransformer.STUDY_CACHE[study_dir]

        # Ищем картинки
        img_paths = glob.glob(os.path.join(study_dir, "*.png"))

        hog_vecs = []
        for p in img_paths:
            img = self._load_image(p)
            hog_vecs.append(self._hog_for_image(img))

        if not hog_vecs:
            raise ValueError(f"No images found in study: {study_dir}")

        # Средний HOG по study
        study_feat = np.mean(hog_vecs, axis=0)

        # Кэшируем
        if self.cache_study:
            StudyHOGTransformer.STUDY_CACHE[study_dir] = study_feat

        return study_feat

    # --------------------------------
    def transform(self, X):
        feats = []
        for study_dir in X:
            feats.append(self._hog_for_study(study_dir))
        return np.vstack(feats)
    
"""
Функция для использования в fit_pipeline_anatomies.
Подготавливает данные для не-xgboost моделей на HOG.
"""
def prepare_data_for_anatomy(base_df: pd.DataFrame, anatomy, get_all=False, is_images=False):
        if get_all:
            X_list, y_list, splits_list = [], [], []
            for anatomy in base_df['anatomy'].unique():
                d = create_andor_return_hog_data(base_df, anatomy=anatomy, is_images=is_images)
                X_list.append(d["X"])
                y_list.append(d["y"])
                splits_list.append(d["splits"])
            data = {
                "X": np.vstack(X_list),
                "y": np.concatenate(y_list),
                "splits": np.concatenate(splits_list),
            }
        else:
            data = create_andor_return_hog_data(base_df, anatomy=anatomy, is_images=is_images)
        X = data['X']
        y = data['y']
        splits = data['splits']

        X_train = X[np.array(splits) == 'train']
        y_train = y[np.array(splits) == 'train']
        X_val = X[np.array(splits) == 'valid']
        y_val = y[np.array(splits) == 'valid']
        return X_train, y_train, X_val, y_val
