import os

# 1) Чтобы не было oversubscription (важно для PCA/BLAS)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from ml.fit import fit_pipeline_anatomies
from ml.hog import prepare_data_for_anatomy
from ml.threshold_classifier import ThresholdClassifier

df_images = pd.read_csv('./data/mura_images_metadata.csv')
# ВАЖНО: здесь ты должен получить df_images так же, как в ноутбуке.
# Например, импортом из твоего модуля:
# from ml.data import load_df_images
# df_images = load_df_images()
#
# Или временно загрузкой из csv:
# import pandas as pd
# df_images = pd.read_csv("...")

def main():
    pipe = Pipeline([
        ("pca", PCA(n_components=30, whiten=True)),
        ("poly", PolynomialFeatures(degree=2)),
        ("clf", ThresholdClassifier(
            model=LogisticRegression(
                max_iter=1000,
                n_jobs=1,
                class_weight="balanced",
            )
        )),
    ])

    N_JOBS = 4  # начни с 4 для стабильности
    result = fit_pipeline_anatomies(
        model_pipeline=pipe,
        param_grid={
            "pca__n_components": [50, 100],
            "clf__model__C": [0.001, 0.01],
            "clf__threshold": np.linspace(0.05, 0.95, 19),
        },
        model_name_base="hog_pca_poly_logreg_pics",
        paths_df=df_images,
        grid_search_params={
            "n_jobs": N_JOBS,
            "verbose": 10,
            "pre_dispatch": "2*n_jobs",
        },
        get_data_for_anatomy=prepare_data_for_anatomy,
        use_all=True,
        is_images=True,
    )

    print(result)

if __name__ == "__main__":
    main()
