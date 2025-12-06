import numpy as np
from skimage import io
from skimage.feature import hog
import pandas as pd


import glob
import os

from sklearn.base import BaseEstimator, TransformerMixin


def compute_study_hog(df_subset, as_gray=True):
        print('updated 1')
        X_feats = []
        y_labels = []
        study_ids = []
        splits = []
        anatomies = []

        for _, row in df_subset.iterrows():
            study_dir = row["path"]
            img_paths = glob.glob(os.path.join(study_dir, "*.png"))

            hog_vecs = []
            for p in img_paths:
                img = io.imread(p, as_gray=as_gray)
                feat = hog(
                    img,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                )
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
    

def prepare_data_for_anatomy(base_df: pd.DataFrame, anatomy, get_all=False):
        if get_all:
            X_list, y_list, splits_list = [], [], []

            for body_part in base_df['anatomy'].unique():
                d = np.load(get_hog_anatomy_filename(body_part), allow_pickle=True)
                X_list.append(d["X"])
                y_list.append(d["y"])
                splits_list.append(d["splits"])
            print("X_list shapes:", [x.shape for x in X_list])
            print("y_list shapes:", [y.shape for y in y_list])
            print("splits_list shapes:", [s.shape for s in splits_list])
            data = {
                "X": np.vstack(X_list),
                "y": np.concatenate(y_list),
                "splits": np.concatenate(splits_list),
            }
        else:
            data = np.load(get_hog_anatomy_filename(anatomy), allow_pickle=True)
        X = data['X']
        y = data['y']
        splits = data['splits']

        X_train = X[np.array(splits) == 'train']
        y_train = y[np.array(splits) == 'train']
        X_val = X[np.array(splits) == 'valid']
        y_val = y[np.array(splits) == 'valid']
        return X_train, y_train, X_val, y_val


def get_hog_anatomy_filename(anatomy):
    return f"hog_{anatomy}.npz"