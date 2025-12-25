# извлечение признаков
from sklearn.decomposition import PCA
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import numpy as np
import pandas as pd
from .data import get_data_path, load_pickle, save_pickle
from scipy.stats import skew, kurtosis, entropy
import cv2
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# Параметры обработки
IMG_SIZE = (224, 224)
IMPLANT_THRESH = 250
MIN_IMPLANT_AREA = 50

# Настройки LBP
LBP_POINTS = 24       # Количество точек
LBP_RADIUS = 3        # Радиус охвата (3 пикселя)
LBP_METHOD = 'uniform'

# Настройки PCA
PCA_COMPONENTS = 50   # Сжимаем визуальные признаки до 50 главных чисел

DATASET_FILENAME_BASE= "feature_dataset_xgb"

def get_dataset_filename():
    return f"{DATASET_FILENAME_BASE}_all.pkl"

def extract_visual_features(img):
    """
    Извлекает "тяжелые" вектора: HOG + LBP
    """
    # 1. HOG (Полный вектор, не среднее!)
    # pixels_per_cell=(32, 32) дает меньше признаков, чем (16,16), но быстрее.
    # feature_vector=True возвращает длинный массив чисел
    hog_vector = hog(img, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=False, feature_vector=True)

    # 2. LBP (Гистограмма текстуры)
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    # Считаем гистограмму LBP значений
    n_bins = LBP_POINTS + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # Объединяем HOG и LBP в один длинный вектор
    return np.concatenate([hog_vector, lbp_hist])
# извлечение статистики и метаданных
def extract_statistical_features(img, anatomy):

    # Нормализация
    flat = img.ravel().astype(np.float32) / 255.0

    feats = {}
    feats['anatomy'] = anatomy # Категория

    # Статистика
    feats['mean'] = np.mean(flat)
    feats['std'] = np.std(flat)
    feats['skew'] = skew(flat)
    feats['kurt'] = kurtosis(flat)
    feats['entropy'] = entropy(np.histogram(flat, bins=64, density=True)[0] + 1e-8)

    # GLCM (на всем изображении для простоты)
    # Для GLCM нужно uint8 0..255
    glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
    feats['glcm_contrast'] = graycoprops(glcm, 'contrast').mean()
    feats['glcm_energy'] = graycoprops(glcm, 'energy').mean()

    return feats

# функция обработки
def process_row(row):
    path = row['path']
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except: return None
    if img is None: return None
    if img.shape != IMG_SIZE: img = cv2.resize(img, IMG_SIZE)

    # Извлекаем ДВА типа признаков
    visual_vec = extract_visual_features(img) # Это пойдет в PCA
    stats_dict = extract_statistical_features(img, row['anatomy']) # Это пойдет напрямую

    return visual_vec, stats_dict

def create_and_return_feature_dataset(images_df: pd.DataFrame):
    """
    Создает и сохраняет датасет признаков для XGboost модели из таблицы изображений.
    Возвращает два массива: визуальные признаки и статистические признаки.
    """
    if (loaded_data := load_pickle(get_data_path(get_dataset_filename()))) is None:
        logger.info("Извлекаем HOG, LBP, GLCM и статистику...")
        visual_data = [] # Список векторов
        stats_data = []  # Список словарей
        labels = []
        # Используем цикл, чтобы удобно разделять потоки данных
        count = 0
        for idx, row in tqdm(images_df.iterrows(), total=len(images_df)):
            res = process_row(row)
            if res:
                vis_vec, stat_dict = res
                visual_data.append(vis_vec)
                stats_data.append(stat_dict)
                labels.append(row['label'])
                count += 1
            else:
                logger.warning("Нет признаков для: %s", row['path'])
        logger.info("Успешно обработано: %s", count)
        # Преобразуем в массивы
        lengths = [len(v) for v in visual_data]
        logger.debug("Уникальные длины визуальных векторов: %s", set(lengths))

        X_visual = np.array(visual_data)
        df_stats = pd.DataFrame(stats_data)
        y = np.array(labels)

        logger.info("Размер визуальной матрицы (до PCA): %s", X_visual.shape)
            # Сначала скейлинг (PCA чувствителен к масштабу)
        scaler = StandardScaler()
        X_visual_scaled = scaler.fit_transform(X_visual)

        # Сам PCA
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        X_pca = pca.fit_transform(X_visual_scaled)

        logger.info("Размер после PCA: %s", X_pca.shape)
        logger.info(
            "Объясненная дисперсия (сколько инфы сохранили): %.2f%%",
            sum(pca.explained_variance_ratio_) * 100,
        )

        # Создаем DataFrame из PCA компонент
        pca_cols = [f'pca_{i}' for i in range(PCA_COMPONENTS)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols)
        # Сбросим индексы, чтобы конкатенация прошла верно
        df_stats.reset_index(drop=True, inplace=True)
        df_pca.reset_index(drop=True, inplace=True)

        X_final = pd.concat([df_stats, df_pca], axis=1)
        X_full = pd.concat([df_stats, df_pca], axis=1)
        X_full['label'] = y  # Временно добавим метку, чтобы удобно фильтровать
        X_full['split'] = images_df['split'].values
        X_full['anatomy'] = images_df['anatomy'].values

        logger.info("Итоговый DataFrame: %s", X_final.shape)
        save_pickle(X_full, get_data_path(get_dataset_filename()))
        return X_full
    else:
        logger.info("Датасет признаков загружен из файла.")
        return loaded_data

"""
Функция для использования в fit_pipeline_anatomies.
Подготавливает данные для XGBoost модели.
как и версия из hog.py, ожидает base_df с путями картинок, а не исследований
"""
def prepare_xgboost_data_for_anatomy(base_df: pd.DataFrame, anatomy: str, get_all=False):
    X_full = create_and_return_feature_dataset(base_df)
    part_df = X_full if get_all else X_full[X_full['anatomy'] == anatomy]
    if len(part_df) < 50:
        return [], [], [], [], 

    X_part = part_df.drop(['label', 'anatomy', 'split'], axis=1)
    y_part = part_df['label']

    if len(y_part.unique()) < 2:
        logger.warning("%s | %s | SKIPPED (1 class)", f"{anatomy:<15}", f"{len(part_df):<8}")
        return [], [], [], [],
    train_mask = part_df['split'] == 'train'
    val_mask = part_df['split'] == 'valid'

    X_train = X_part[train_mask]
    y_train = y_part[train_mask]
    X_val  = X_part[val_mask]
    y_val  = y_part[val_mask]
    return X_train, y_train, X_val, y_val
