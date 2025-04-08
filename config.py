import os
import sys

ACCEPTED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif','.webp']

PROJECT_PATH = '/home/n230ai/Documentos/aplicaciones/n230ai_ocr_v2/'
INPUT_FILES = os.path.join(PROJECT_PATH,'files')
IMAGES_INPUT = os.path.join(PROJECT_PATH,'images_input')
IMAGES_RESTORED = os.path.join(PROJECT_PATH,'images_restored')
TMP_DIR = os.path.join(PROJECT_PATH,'tmp_images')
KNOWLEDGE = os.path.join(PROJECT_PATH,"knowledge")

# Añadir el directorio del paquete convert y restormer al path
CONVERTER = os.path.join(PROJECT_PATH, 'convert')
RESTOMER = os.path.join(PROJECT_PATH, 'restormer')

sys.path.append(CONVERTER)
sys.path.append(RESTOMER)
