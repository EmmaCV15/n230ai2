from config import PROJECT_PATH
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_name = "Qwen2.5-VL-7B-Instruct"
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

folder_path = os.path.join(PROJECT_PATH,f'qwen_ocr/Qwen25-VL-7B-Instruct')
# Verificar si la carpeta existe, si no, crearla
if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Creando la carpeta: {folder_path}')

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

try:
    model.save_pretrained(folder_path)
    processor.save_pretrained(folder_path)
    print(f'Modelo guardado en la carpeta: {folder_path}')
except:
    raise ValueError('No se pudo guardar el modelo.')