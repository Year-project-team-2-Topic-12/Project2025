import pickle
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import os
import glob
import numpy as np
from skimage import io
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

def save_model_data(model, name):
    with open(name, "wb") as f:
        pickle.dump(model, f)

def load_model_data(name):
    try:
        with open(name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File {name} not found.")
        return None
def print_metrics(y_train, y_pred_train, y_val, y_pred_val):
    acc = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_train, y_pred_train, weights='quadratic')
    print(f"TRAIN METRICS: Accuracy={acc:.3f}, F1_Abnormal={f1:.3f}, Cohen_Kappa={kappa:.3f}")
    train_data = {'accuracy': acc, 'f1': f1, 'kappa': kappa}
    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_val, y_pred_val, weights='quadratic')
    print(f"VALID METRICS: Accuracy={acc:.3f}, F1_Abnormal={f1:.3f}, Cohen_Kappa={kappa:.3f}")
    valid_data = {'accuracy': acc, 'f1': f1, 'kappa': kappa}
    return {'train': train_data, 'valid': valid_data}

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
