import ollama
import os
import gradio as gr

from utils.ollama_utils import run_ollama_model,start_ollama_server,kill_ollama_server
from config import KNOWLEDGE
from app import ocr_result


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

def inference(text, template_file):

    model_name = 'qwen2.5-max:14b'

    # Restart Ollama
    start_ollama_server()

    # Load Model
    run_ollama_model(model_name)

    # Get Knowledge
    # template_file = 'template_propiedad.txt'
    system_prompt, instructions, template = get_knowledge(template_file=template_file)

    # Prepare Prompt
    user_prompt = instructions + text + template

    # Make inference
    stream = ollama.chat(model=model_name,
                  messages=[
                      {
                          'role': 'system',
                          'content': system_prompt,
                      },
                      {
                          'role': 'user',
                          'content': user_prompt,
                      },
                  ],
                  options={
                      "temperature": 0.1,
                      "top_p": 0.95,
                      "top_k":40,

                  },
                  stream=True,
                  )

    text = ''
    for chunk in stream:
        text += chunk['message']['content']
        yield text

def get_template_path(choice):

    options_catalogue = [
        {'antecedente de propiedad': 'template_propiedad.txt'},
        {},
        {},
    ]

    file_name = options_catalogue[choice]

    return file_name

def read_file(file_name):
    with open(file_name, "r") as f:
        return f.read()

with (gr.Blocks(css="footer {display: none !important}", fill_width=True, title="N230-OCR") as app):
    # Components
    with gr.Row():

        with gr.Column(scale=2):
            input_text = gr.Textbox(
                inputs= ocr_result,
                label='Texto Original',
                lines=25,
                max_lines=25,
                interactive=True,
                autoscroll=True,
                show_copy_button=True,
            )

        with gr.Column(scale=2):
            response = gr.Textbox(
                label='Texto Generado',
                lines=25,
                max_lines=25,
                interactive=True,
                autoscroll=True,
                show_copy_button=True,
            )

    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(visible=False)

            task = gr.Dropdown(
                container=False,
                # info='Seleccione una tarea',
                label='Tarea',
                show_label=False,
                choices=['antecedente de propiedad']
            )

        with gr.Column(scale=1):
            inference_btn = gr.Button("Generar")

        with gr.Column(scale=1):
            download_btn = gr.DownloadButton("Descargar")


    # Events
    file.upload(
        fn=read_file,
        inputs=[file],
        outputs=[input_text]
    )

    inference_btn.click(
        fn=inference,
        inputs=[input_text, task],
        outputs=response
    )

if __name__ == '__main__':

    app.launch()