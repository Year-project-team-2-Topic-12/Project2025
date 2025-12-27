from dotenv import load_dotenv
import os

load_dotenv()
from pathlib import Path

# Идём вверх, пока не найдём pyproject.toml
ROOT_DIR = Path(__file__).resolve()
while not (ROOT_DIR / "pyproject.toml").exists():
    if ROOT_DIR.parent == ROOT_DIR:
        raise RuntimeError("Cannot find project root (pyproject.toml)")
    ROOT_DIR = ROOT_DIR.parent

ROOT_DIR = str(ROOT_DIR)
MODELS_PATH = os.path.join(ROOT_DIR, os.getenv('MODELS_PATH', 'models/'))
DATA_PATH = os.path.join(ROOT_DIR, os.getenv('DATA_PATH', 'data/'))
RESULTS_PATH = os.path.join(ROOT_DIR, os.getenv('RESULTS_PATH', 'results/'))
DATA_ROOT = os.path.join(ROOT_DIR, os.getenv('DATA_ROOT', 'MURA-v1.1-resized-224x224'))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '224'))
SELECTED_MODEL = os.getenv('SELECTED_MODEL', 'hog_pca_poly_logreg')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin')
LOGS_FILE = os.getenv('LOG_FILE', 'logs/ml_service.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
SECRET_KEY = os.getenv('SECRET_KEY', 'SUPER_SECRET_KEY_CHANGE_ME')
SEED = int(os.getenv('SEED', '42'))