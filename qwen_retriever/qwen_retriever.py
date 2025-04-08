import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
from config import PROJECT_PATH, KNOWLEDGE
import torch
from accelerate import Accelerator

def check_memory():
    print(f"Memoria asignada: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memoria máxima asignada: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Pila de memoria: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Pila de memoria máxima: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

def get_knowledge(template_file: str='template_propiedad.txt'):
    try:
        files = ['system_prompt.txt', 'instructions.txt', template_file]
        contents = []
        for file in files:
            file_path = os.path.join(KNOWLEDGE, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    contents.append(f.read())
            else:
                print(f"Advertencia: {file} no encontrado.")
                contents.append("")
        return tuple(contents)
    except Exception as e:
        print(f"Error al leer archivos de conocimiento: {e}")
        return "", "", ""

def load_model(model_path: str = ''):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. Asegúrate de tener una GPU compatible.")

    accelerator = Accelerator()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    max_memory = {i: "22GB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
            max_memory=max_memory
        )
        model, tokenizer = accelerator.prepare(model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

def inference(model_path: str = '', text: str = ''):

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    model, tokenizer = load_model(model_path)
    if model is None or tokenizer is None:
        return "Error: No se pudo cargar el modelo."

    system_prompt, instrucciones, template = get_knowledge('template_propiedad.txt')

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instrucciones + '\n' + text + '\n' + template}
    ]

    inputext = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([inputext], return_tensors="pt").to(model.device)

    # streamer = TextStreamer(tokenizer, skip_prompt=True)

    try:

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            repetition_penalty= 1.50,
            pad_token_id=tokenizer.eos_token_id,
        )

        if generated_ids is None or len(generated_ids) == 0:
            print("Error: No se generaron tokens.")
            return ""

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Split assistant answer only
        if 'assistant' in response:
            response = response.split('assistant')[-1].strip()

        return response

    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        return ""

if __name__ == '__main__':

    # Clean caché
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    text_path = os.path.join(PROJECT_PATH, 'outputs', 'prueba_compraventa.txt')
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"No se encontró el archivo: {text_path}")

    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    options = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct-1M", "Qwen2.5-32B-Instruct-AWQ","DeepSeek-R1-Distill-Qwen-14B"]

    for model_name in options:
        if model_name == 'DeepSeek-R1-Distill-Qwen-14B':
            model_path = os.path.join(PROJECT_PATH, 'qwen_retriever', model_name)
            response = inference(text=text, model_path=model_path)
            print(response)
            break
        else:
            continue
