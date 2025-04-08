from config import PROJECT_PATH
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# models = [
# ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
# ("Qwen2.5-14B-Instruct-1M", "Qwen/Qwen2.5-14B-Instruct-1M"),
# ("Qwen2.5-32B-Instruct-AWQ", "Qwen/Qwen2.5-32B-Instruct-AWQ")
# ]

models = [("DeepSeek-R1-Distill-Qwen-14B","deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")]

for model_name, model_path in models:
    folder_path = os.path.join(PROJECT_PATH, 'qwen_retriever', model_name)

    # Crear la carpeta si no existe
    os.makedirs(folder_path, exist_ok=True)
    print(f'Usando la carpeta: {folder_path}')

    try:
        # Descargar y guardar el tokenizer primero
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(folder_path)
        print(f'Tokenizer de {model_name} guardado en la carpeta: {folder_path}')

        # Descargar y guardar el modelo
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Se fuerza float16 directamente
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        model.save_pretrained(folder_path)
        print(f'Modelo {model_name} guardado en la carpeta: {folder_path}')

    except Exception as e:
        print(f'Error al descargar el modelo {model_name}: {e}')