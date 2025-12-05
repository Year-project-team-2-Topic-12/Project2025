import pickle
from IPython.display import display
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import os
import glob
import numpy as np
from skimage import io
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from sklearn.metrics import make_scorer, cohen_kappa_score

kappa_scorer = make_scorer(cohen_kappa_score)

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
def print_return_metrics(y_train, y_pred_train, y_val, y_pred_val, y_pred_train_decfun, y_pred_val_decfun):
    acc = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_train_decfun)
    print(f"TRAIN METRICS: Accuracy={acc:.3f}, F1_Abnormal={f1:.3f}, Cohen_Kappa={kappa:.3f}, ROC_AUC={roc_auc_train:.3f}")
    train_data = {'accuracy': acc, 'f1': f1, 'kappa': kappa, 'roc_auc': roc_auc_train}
    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_val, y_pred_val, weights='quadratic')
    roc_auc_val = roc_auc_score(y_val, y_pred_val_decfun)
    print(f"VALID METRICS: Accuracy={acc:.3f}, F1_Abnormal={f1:.3f}, Cohen_Kappa={kappa:.3f}, ROC_AUC={roc_auc_val:.3f}")
    valid_data = {'accuracy': acc, 'f1': f1, 'kappa': kappa, 'roc_auc': roc_auc_val}
    return {'train': train_data, 'valid': valid_data}

def get_hog_anatomy_filename(anatomy):
    return f"hog_{anatomy}.npz"

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


def fit_hog_pipeline_bodyparts(pipe_bodypart: Pipeline, param_grid: dict, model_name_base: str, base_df: pd.DataFrame, grid_search_params={}, use_all=False):
    resulting_data = {
        'anatomy': [],
        'train_kappa': [],
        'valid_kappa': [],
        'train_accuracy': [],
        'valid_accuracy': [],
        'train_f1': [],
        'valid_f1': [],
        'best_params': [],
        'train_roc_auc': [],
        'valid_roc_auc': [],
        'model_name_base': [],
        'fit_time_seconds': [],
        'fit_time_std': [],
    }
    # models = []
    results_fname = f"results_{model_name_base}.csv"
    results_df = None
    try:
        results_df = pd.read_csv(results_fname)
    except FileNotFoundError:
        pass
    if results_df is not None:
        return results_df

    def train_model_for_bodypart(body_part: str, data):
        print(f"\nОбработка анатомии: {body_part}")
        model_fname = f"model_{model_name_base}_{body_part}.pkl"
        X = data['X']
        y = data['y']
        splits = data['splits']

        X_train = X[np.array(splits) == 'train']
        y_train = y[np.array(splits) == 'train']
        X_val = X[np.array(splits) == 'valid']
        y_val = y[np.array(splits) == 'valid']
        if not(grid_search := load_model_data(model_fname)):
            print(f"Нет сохранённой - обучаем модель {model_name_base} для анатомии {body_part}")
            print("len X_train:", len(X_train))
            print("len X_val:", len(X_val))
            grid_params = {
                'param_grid': param_grid, 'scoring': kappa_scorer, 'cv': 5, 'n_jobs': -1, **grid_search_params
            }

            grid_search = GridSearchCV(pipe_bodypart, **grid_params)
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            fit_time = time.time() - start_time
            save_model_data(grid_search, model_fname)
            print("Время обучения grid search (сек):", fit_time)
        best_model = grid_search.best_estimator_
        print("Best params:", grid_search.best_params_)
        best_idx = grid_search.best_index_
        fit_time_best = grid_search.cv_results_["mean_fit_time"][best_idx]
        fit_time_best_std = grid_search.cv_results_["std_fit_time"][best_idx]
        print("Mean fit time:", fit_time_best)
        print("Std fit time:", fit_time_best_std)
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        y_pred_train_decfun = best_model.decision_function(X_train)
        y_pred_val_decfun = best_model.decision_function(X_val)

        resulting_data['model_name_base'].append(model_name_base)
        resulting_data['fit_time_seconds'].append(fit_time_best)
        resulting_data['fit_time_std'].append(fit_time_best_std)

        metrics = print_return_metrics(y_train,
                                y_pred_train,
                                y_val,
                                y_pred_val,
                                y_pred_train_decfun=y_pred_train_decfun,
                                y_pred_val_decfun=y_pred_val_decfun)

        resulting_data['anatomy'].append(body_part)
        resulting_data['train_kappa'].append(metrics['train']['kappa'])
        resulting_data['valid_kappa'].append(metrics['valid']['kappa'])
        resulting_data['train_accuracy'].append(metrics['train']['accuracy'])
        resulting_data['valid_accuracy'].append(metrics['valid']['accuracy'])
        resulting_data['train_f1'].append(metrics['train']['f1'])
        resulting_data['valid_f1'].append(metrics['valid']['f1'])
        resulting_data['best_params'].append(grid_search.best_params_)
        resulting_data['train_roc_auc'].append(metrics['train']['roc_auc'])
        resulting_data['valid_roc_auc'].append(metrics['valid']['roc_auc'])
            # models.append((body_part, best_model))

    print(f"Обучаем модель {model_name_base} по всему датасету")
    if use_all:
        X_list, y_list, splits_list = [], [], []

        for body_part in base_df['anatomy'].unique():
            d = np.load(get_hog_anatomy_filename(body_part), allow_pickle=True)
            X_list.append(d["X"])
            y_list.append(d["y"])
            splits_list.append(d["splits"])
        print("X_list shapes:", [x.shape for x in X_list])
        print("y_list shapes:", [y.shape for y in y_list])
        print("splits_list shapes:", [s.shape for s in splits_list])
        data_all = {
            "X": np.vstack(X_list),
            "y": np.concatenate(y_list),
            "splits": np.concatenate(splits_list),
        }
        print("Any empty X?", any(x.shape[0] == 0 for x in X_list))
        print("Unique splits:", set(np.concatenate(splits_list)))
        train_model_for_bodypart("ALL_anatomies", data_all)
    else: 
        for body_part in base_df['anatomy'].unique():
            data = np.load(get_hog_anatomy_filename(body_part), allow_pickle=True)
            train_model_for_bodypart(body_part, data)
        
    print(f"Обучаем модель {model_name_base} по анатомиям")

    results_df =  pd.DataFrame(resulting_data)
    results_df.to_csv(results_fname, index=False)
    return results_df