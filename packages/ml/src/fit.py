from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, make_scorer, roc_auc_score
import optuna
from optuna.integration import OptunaSearchCV
from typing import Any, Callable, Protocol, TypedDict, Unpack, overload
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from .data import load_model_pipeline, load_model_results, save_model_pipeline, save_model_results, get_data_path, get_full_model_name
from ml.env import SEED

ArrayLike = np.ndarray | pd.Series | pd.DataFrame

logger = logging.getLogger(__name__)

class GetDataKwargs(TypedDict, total=False):
    is_images: bool | None
    study_aggs: dict[str, Callable[[np.ndarray], np.ndarray]] | None
    hog_n_jobs: int | None
    extract_func: Callable[[str], dict[str, Any]] | None

class GetDataFunction(Protocol):
    @overload
    def __call__(
        self,
        base_df: pd.DataFrame,
        anatomy: str | None = None,
        get_all: bool = False, # Если True, то игнорируем anatomy,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: ... # X_train, y_train, X_val, y_val

    @overload
    def __call__(
        self,
        base_df: pd.DataFrame,
        anatomy: str | None = None,
        get_all: bool = False, # Если True, то игнорируем anatomy,
        **kwargs: Unpack[GetDataKwargs],
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: ... # X_train, y_train, X_val, y_val

OPTUNA_DEFAULT_PARAMS = {
    'n_trials': 50,
    'n_jobs': 8,
    'cv': 5,
    'refit': True,
    'random_state': SEED,
    # "show_progress_bar": True,
}

def get_data_for_anatomy_default(base_df: pd.DataFrame, anatomy: str | None = None, get_all: bool = False, **kwargs) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    # Заглушка по умолчанию
    return np.array([]), np.array([]), np.array([]), np.array([])

kappa_scorer = make_scorer(cohen_kappa_score)
"""
Функция для обучения модели по анатомиям с использованием GridSearchCV.
@param model_pipeline: sklearn Pipeline, содержащий препроцессинг и модель.
@param model_name_base: Базовое имя модели для сохранения результатов и модели.
@param paths_df: DataFrame с путями к данным и метками.
@param param_grid: Словарь с параметрами для GridSearchCV.
@param grid_search_params: Дополнительные параметры для GridSearchCV. Параметры по умолчанию: cv=5, scoring=kappa_scorer, n_jobs=-1.
@param use_all: Если True, обучаем модель на всех данных сразу.
@param use_decision_function: Если True, используем decision_function для ROC AUC, иначе используем predict_proba.
@param get_data_for_anatomy: Функция для получения данных по анатомии.
@param anatomy: Если указано, обучаем модель только для этой анатомии.
@return: DataFrame с результатами обучения модели.
"""
def fit_pipeline_anatomies(
        model_pipeline: Pipeline,
        model_name_base: str,
        paths_df: pd.DataFrame,
        param_grid: dict = {},
        grid_search_params={},
        use_all=False,
        use_decision_function=True,
        get_data_for_anatomy: GetDataFunction  = get_data_for_anatomy_default,
        only_anatomy: str | None = None,
        is_optuna: bool = False,
        params_get_data: dict = {},
        ):

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
    results_df = load_model_results(model_name_base, use_all=use_all)
    if results_df is not None:
        logger.info("Загружены результаты модели %s из файла.", model_name_base)
        return results_df

    def train_model_for_anatomy(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, anatomy: str | None = None):
        logger.info("Обработка анатомии: %s", anatomy if anatomy else "ALL")
        if not(grid_search := load_model_pipeline(model_name_base, anatomy=anatomy)):
            logger.info("Нет сохранённой - обучаем модель %s для анатомии %s", model_name_base, anatomy)
            logger.debug("len X_train: %s", len(X_train))
            logger.debug("len X_val: %s", len(X_val))
            if is_optuna:
                study_name = model_name_base if anatomy is None else get_full_model_name(model_name_base, anatomy)
                storage_name = f"sqlite:///{get_data_path(f'{study_name}.db')}"
                study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
                optuna_params = {
                    **OPTUNA_DEFAULT_PARAMS,
                    'param_distributions': param_grid,
                    'scoring': kappa_scorer,
                    'study': study,
                    **grid_search_params,
                }
                grid_search = OptunaSearchCV(model_pipeline, **optuna_params)
            else:
                grid_params = {
                    'param_grid': param_grid, 'scoring': kappa_scorer, 'cv': 5, 'n_jobs': -1, **grid_search_params
                }
                grid_search = GridSearchCV(model_pipeline, **grid_params)
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            fit_time = time.time() - start_time
            save_model_pipeline(grid_search, model_name_base, anatomy=anatomy)
            logger.info("Время обучения grid search (сек): %s", fit_time)
        else:
            logger.info("Загружена сохранённая модель %s для анатомии %s", model_name_base, anatomy)
        best_model = grid_search.best_estimator_
        logger.info("Best params: %s", grid_search.best_params_)
        best_idx = grid_search.best_index_
        fit_time_best = grid_search.cv_results_["mean_fit_time"][best_idx]
        fit_time_best_std = grid_search.cv_results_["std_fit_time"][best_idx]
        logger.info("Mean fit time: %s", fit_time_best)
        logger.info("Std fit time: %s", fit_time_best_std)
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        if use_decision_function:
            y_pred_train_decfun = best_model.decision_function(X_train)
            y_pred_val_decfun = best_model.decision_function(X_val)
        else:
            y_pred_train_decfun = best_model.predict_proba(X_train)[:, 1]
            y_pred_val_decfun = best_model.predict_proba(X_val)[:, 1]

        resulting_data['model_name_base'].append(model_name_base)
        resulting_data['fit_time_seconds'].append(fit_time_best)
        resulting_data['fit_time_std'].append(fit_time_best_std)

        metrics = print_return_metrics(y_train,
                                y_pred_train,
                                y_val,
                                y_pred_val,
                                y_pred_train_decfun=y_pred_train_decfun,
                                y_pred_val_decfun=y_pred_val_decfun)

        resulting_data['anatomy'].append(anatomy)
        resulting_data['train_kappa'].append(metrics['train']['kappa'])
        resulting_data['valid_kappa'].append(metrics['valid']['kappa'])
        resulting_data['train_accuracy'].append(metrics['train']['accuracy'])
        resulting_data['valid_accuracy'].append(metrics['valid']['accuracy'])
        resulting_data['train_f1'].append(metrics['train']['f1'])
        resulting_data['valid_f1'].append(metrics['valid']['f1'])
        resulting_data['best_params'].append(grid_search.best_params_)
        resulting_data['train_roc_auc'].append(metrics['train']['roc_auc'])
        resulting_data['valid_roc_auc'].append(metrics['valid']['roc_auc'])

    if use_all:
        logger.info("Обучаем модель %s по всему датасету", model_name_base)
        X_train, y_train, X_val, y_val = get_data_for_anatomy(
            paths_df,
            "ALL_anatomies",
            get_all=True,
            **params_get_data,
        )
        train_model_for_anatomy(X_train, y_train, X_val, y_val)
    else:
        logger.info("Обучаем модель %s по анатомиям", model_name_base)
        for anatomy in paths_df['anatomy'].unique():
            if only_anatomy and anatomy != only_anatomy:
                continue
            logger.info("Обучаем модель по анатомии %s", anatomy)
            X_train, y_train, X_val, y_val = get_data_for_anatomy(
                paths_df,
                anatomy,
                **params_get_data,
            )
            train_model_for_anatomy(X_train, y_train, X_val, y_val, anatomy=anatomy)

    results_df =  pd.DataFrame(resulting_data)
    save_model_results(results_df, model_name_base, use_all=use_all)
    return results_df


def print_return_metrics(y_train, y_pred_train, y_val, y_pred_val, y_pred_train_decfun, y_pred_val_decfun):
    acc = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_train_decfun)
    logger.info(
        "TRAIN METRICS: Accuracy=%.3f, F1_Abnormal=%.3f, Cohen_Kappa=%.3f, ROC_AUC=%.3f",
        acc,
        f1,
        kappa,
        roc_auc_train,
    )
    train_data = {'accuracy': acc, 'f1': f1, 'kappa': kappa, 'roc_auc': roc_auc_train}
    acc = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_pred_val_decfun)
    logger.info(
        "VALID METRICS: Accuracy=%.3f, F1_Abnormal=%.3f, Cohen_Kappa=%.3f, ROC_AUC=%.3f",
        acc,
        f1,
        kappa,
        roc_auc_val,
    )
    valid_data = {'accuracy': acc, 'f1': f1, 'kappa': kappa, 'roc_auc': roc_auc_val}
    return {'train': train_data, 'valid': valid_data}
    if is_images is not None and "is_images" not in params_get_data:
        params_get_data = {**params_get_data, "is_images": is_images}
