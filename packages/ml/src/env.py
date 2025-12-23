from dotenv import load_dotenv
import os

load_dotenv()

MODELS_PATH = os.getenv('MODELS_PATH', 'models/')
DATA_PATH = os.getenv('DATA_PATH', 'data/')
RESULTS_PATH = os.getenv('RESULTS_PATH', 'results/')
DATA_ROOT = os.getenv('DATA_ROOT', 'MURA-v1.1-resized-224x224')
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '224'))
SELECTED_MODEL = os.getenv('SELECTED_MODEL', 'hog_pca_poly_logreg')