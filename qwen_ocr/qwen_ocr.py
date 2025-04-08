import os
import time

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
from threading import Thread
import torch
from PIL import Image
import datetime

# Custom functions
from config import PROJECT_PATH
from utils.gpu_utils import *
from utils.ollama_utils import *
from utils.images_utils import calculate_density

def load_model(model_name='Qwen25-VL-7B-Instruct'):

    MODEL_PATH: str = os.path.join(PROJECT_PATH, 'qwen_ocr', model_name)

    processor = AutoProcessor.from_pretrained(MODEL_PATH,
                                                      trust_remote_code=True,
                                                      )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    print(f'Model loaded: {model_name} Device: {model.device}')

    return model, processor

def qwen_ocr(uploaded_file=None, gallery=None):

    # Get user id
    parts = str(uploaded_file).split(os.sep)
    user_id = parts[-2]

    print('-'*100)
    print(f'OCR Image to text process:')
    print('-' * 100)
    current_date = datetime.datetime.now()
    # Formatear la fecha y hora en el formato deseado
    current_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Date: {current_date}')
    print(f'User id: {user_id}')
    print(f'Gallery: {len(gallery)} {gallery}')

    # Clean GPU cache
    torch.cuda.empty_cache()

    # Load model
    model_name = 'Qwen25-VL-7B-Instruct'
    model, processor = load_model(model_name=model_name)

    start_time = time.perf_counter()

    buffer = ''
    for idx, (image_path, page_number) in enumerate(gallery):

        page_start_time = time.perf_counter()

        gr.Info(f'Leyendo imagen {idx + 1} de {len(gallery)}.', duration=1)

        # Get density
        image = Image.open(image_path)
        density = calculate_density(image)

        # Load image
        loaded_image = load_image(image)

        system_prompt = """
            [EN]
            You are an accurate and precise expert in text transcription from scanned documents and images.
            [ES]
            Eres un experto preciso y exacto en la transcripción de texto de documentos escaneados e imágenes.
            [CH]
            您是一位在扫描文件和图像中文本转录方面准确且精确的专家。
            """

        user_prompt = """
            [EN]
             Instructions:
             - Improve the image contrast and apply sharpening to make the text clearer.
             - Split the image into smaller, more manageable sections and apply OCR individually to each section.
             - Try and apply different custom configurations in OCR to improve text recognition.
             - Extract the whole body textual content from the attached image with the highest precision and accuracy possible.
             - Omit the headers, page numbers and footers.
             - Do not extract consecutive dashes when there are more than three in a row (e.g., - - - -).
             - Ensure to not invent or change any word; the accuracy and precision must be 100%.
             - Return text in Plain Text in Spanish, without additional comments and notes.
            [ES]
            Instrucciones:
             - Mejora el contraste de la imagen y aplica nitidez para que el texto sea más claro.
             - Divide la imagen en secciones más pequeñas y manejables y aplica OCR individualmente a cada sección.
             - Intenta aplicar diferentes configuraciones personalizadas en OCR para mejorar el reconocimiento de texto.
             - Extrae todo el cuerpo del contenido textual de la imagen adjunta con la mayor precisión y exactitud posible.
             - Omite los encabezados, números de página y notas al pie.
             - Omite los guiones medios cuando aparecen más de tres seguidos (por ejemplo, - - - ).
             - Asegúrate de no inventar ni cambiar ninguna palabra; la precisión y exactitud deben ser del 100%.
             - Devuelve el texto extraído en Texto Plano en español, sin comentarios ni notas adicionales.
             [CH]
             指示：
             - 提高图像对比度并应用锐化处理，使文本更加清晰。
             - 将图像分割成更小、更易于管理的部分，并对每个部分单独应用OCR。
             - 尝试应用不同的OCR自定义配置以提高文本识别效果。
             - 从附带的图像中提取所有文本内容，确保精度和准确度达到最高。
             - 删除页眉、页码和页脚。
             - 不要删除连续出现的多个中划线（例如，- – – ）。
             - 确保不添加或更改任何词语；准确度和精密度必须达到100%。
             - 返回西班牙语的纯文本，不包含额外的注释和备注。
             """
        # - Devuelve el texto en Texto Plano en español, sin comentarios ni notas adicionales.

        # Prepare messages for the model
        messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": loaded_image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

        # Apply chat template and process inputs
        prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = processor(
                text=[prompt],
                images=[loaded_image],
                padding=True,
                return_tensors="pt",
            )
        inputs.to("cuda")
        # ---------------------
        # Set up streamer for real-time output
        streamer = TextIteratorStreamer(processor,
                                            skip_prompt=True,
                                            skip_special_tokens=True
                                            )

        generation_kwargs = dict(inputs,
                                     streamer=streamer,
                                     max_new_tokens=2048
                                     )
        # Start generation in a separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the output
        if idx == 0:
            yield f"Leyendo..."

        for new_text in streamer:
                buffer += new_text
                time.sleep(0.01)
                # Clean GPU cache
                torch.cuda.empty_cache()

                yield buffer
        buffer += f'\n\n'

        page_end_time = time.perf_counter()
        page_time = page_end_time - page_start_time

        for device in gpu_checkup():
            print(f'Read: {idx+1} of {len(gallery)}: Page number: {page_number} Density: {density:.3f}% Size: {image.size} Used VRAM: {device.memoryUsed:.0f}GB ({device.memoryUtil * 100:.2f}%) Free: {device.memoryFree}GB Page time: {page_time:.2f} seg')

    # Delete model
    del model

    # Clean GPU cache
    torch.cuda.empty_cache()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    average_time = total_time / len(gallery)

    print(f'Total read pages: {len(gallery)} Total Time: {total_time:.2f} seg Average time: {average_time:.2f} seg/pag')
    print('-'*100)

    # User message
    gr.Info("Lectura terminada.", duration=2)

    return buffer