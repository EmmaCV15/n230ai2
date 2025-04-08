import os.path
import glob
import time
from queue import Queue
import gradio as gr
import torch.cuda
import datetime
import queue

# For Retrieval
import ollama
from utils.ollama_utils import run_ollama_model,start_ollama_server,kill_ollama_server

# For OCR
from converter.converter import *
from restormer.restormer import images_restoration
from qwen_ocr.qwen_ocr import qwen_ocr
from config import INPUT_FILES, IMAGES_INPUT, IMAGES_RESTORED, KNOWLEDGE, ACCEPTED_EXTENSIONS
from utils.ollama_utils import *
from utils.gpu_utils import *

# FOR OCR
convert_queue = queue.Queue()
restore_queue = queue.Queue()
ocr_queue = queue.Queue()

def queue_manager(uploaded_file:str=None, step:int=1):

    parts = str(uploaded_file).split(os.sep)
    user_id = parts[-2]

    """Maneja la posición del usuario en cada cola"""
    # Conversion Queue
    if step == 1:
        print('-'*100)
        print('Append to Conversion queue')
        print(f'User id: {user_id}')

        convert_queue_list = list(dict.fromkeys(convert_queue.queue))

        if user_id not in convert_queue_list:
            convert_queue.put(user_id)
            print(f'Append to queue: {user_id}.')
        else:
            print(f'Already in queue: {user_id}.')

        # Get updated list
        convert_queue_list = list(dict.fromkeys(convert_queue.queue))
        # Get user position
        position = convert_queue_list.index(user_id) + 1 if user_id in convert_queue_list else ocr_queue.qsize()
        # Get Size
        total_in_queue = convert_queue.qsize()

        print(f'Current Conversion queue {total_in_queue} users: {convert_queue_list}')
        print(f'-'*100)

        # if total_in_queue > 1:
        #     gr.Info(f'turno: {position} de {total_in_queue}', duration=0)
        return f'''<p style="font-size: 11px; text-align: right">Turno {position} de {total_in_queue}</p>'''

    # Restoration Queue
    elif step == 2:
        print('-' * 100)
        print('Append to Restoration queue')
        print(f'User id: {user_id}')

        restore_queue_list = list(dict.fromkeys(restore_queue.queue))

        if user_id not in restore_queue_list:
            restore_queue.put(user_id)
            print(f'Append to queue: {user_id}.')
        else:
            print(f'Already in queue: {user_id}.')

        # Get updated list
        restore_queue_list = list(dict.fromkeys(restore_queue.queue))
        position = restore_queue_list.index(user_id) + 1 if user_id in restore_queue_list else restore_queue.qsize()
        total_in_queue = restore_queue.qsize()

        print(f'Current Restore queue {total_in_queue} users: {restore_queue_list}')
        print(f'-'*100)

        # if total_in_queue > 1:
        #     gr.Info(f'turno: {position} de {total_in_queue}', duration=0)
        return f'''<p style="font-size: 11px; text-align: right">Turno {position} de {total_in_queue}</p>'''

    # OCR Queue
    elif step == 3:
        print('-' * 100)
        print('Append to OCR queue')
        print(f'User id: {user_id}')

        ocr_queue_list = list(dict.fromkeys(ocr_queue.queue))

        if user_id not in ocr_queue_list:
            ocr_queue.put(user_id)
            print(f'Append to queue: {user_id}.')
        else:
            print(f'Already in queue: {user_id}.')

        # Get updated list
        ocr_queue_list = list(dict.fromkeys(ocr_queue.queue))
        position = ocr_queue_list.index(user_id) + 1 if user_id in ocr_queue_list else ocr_queue.qsize()
        total_in_queue = ocr_queue.qsize()

        print(f'Current OCR queue {total_in_queue} users: {ocr_queue_list}')
        print(f'-'*100)

        # if total_in_queue > 1:
        #     gr.Info(f'turno: {position} de {total_in_queue}', duration=0)
        return f'''<p style="font-size: 11px; text-align: right">Turno {position} de {total_in_queue}</p>'''

def complete_queue(uploaded_file, step):

    global convert_queue
    global restore_queue
    global ocr_queue

    parts = str(uploaded_file).split("/")
    user_id = parts[-2]

    # Conversion Queue
    if step == 1:
        print('-' * 100)
        print('Remove from Conversion queue')
        print(f'User id: {user_id}')

        # Make temp queue objet
        temp_queue = Queue()
        user_found = False
        while not convert_queue.empty():
            current_user = convert_queue.get()

            if user_id != current_user:
                temp_queue.put(current_user)

            else:
                user_found = True
                print(f'Removed from queue: {user_id}')

        if not user_found:
            print(f'User not found in queue {user_id}')

        convert_queue = temp_queue
        # Get updated size
        total_in_queue = convert_queue.qsize()
        # Get updated list
        convert_queue_list = list(dict.fromkeys(convert_queue.queue))

        print(f'Current convert queue {total_in_queue} users: {convert_queue_list}')
        print('-'*100)

    # Restore queue
    elif step == 2:
        print('-' * 100)
        print('Remove from Restore queue')
        print(f'User id: {user_id}')

        temp_queue = Queue()
        user_found = False

        while not restore_queue.empty():

            current_user = restore_queue.get()

            if user_id != current_user:
                temp_queue.put(current_user)
            else:
                user_found = True
                print(f'Removed from queue: {user_id}')

        if not user_found:
            print(f'User not found in queue {user_id}')

        restore_queue = temp_queue
        # Get updated size
        total_in_queue = restore_queue.qsize()
        # Get updated list
        restore_queue_list = list(dict.fromkeys(restore_queue.queue))

        print(f'Current Restore queue {total_in_queue} users: {restore_queue_list}')
        print('-'*100)

    elif step == 3:
        print('-' * 100)
        print('Remove from OCR queue')
        print(f'User id: {user_id}')

        temp_queue = Queue()
        user_found = False

        while not ocr_queue.empty():

            current_user = ocr_queue.get()

            if user_id != current_user:
                temp_queue.put(current_user)
            else:
                user_found = True
                print(f'Removed from queue: {user_id}')

        if not user_found:
            print(f'User not found in queue {user_id}')

        ocr_queue = temp_queue
        # Get updated size
        total_in_queue = ocr_queue.qsize()
        # Get updated list
        ocr_queue_list = list(dict.fromkeys(ocr_queue.queue))

        print(f'Current 0CR queue {total_in_queue} users: {ocr_queue_list}')
        print('-'*100)

# Función para limpiar todos los directorios temporales
def clean_files():
    """ Eliminar todos los archivos temporales de los siguientes directorios:"""

    directories = [INPUT_FILES, IMAGES_INPUT, IMAGES_RESTORED]

    # Por cada directorio, eliminar todos los archivos
    removed_files = []
    removed_folders = []
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(path):  # Verificar si es un archivo
                        os.unlink(path)  # Eliminar el archivo
                        removed_files.append(path)

                    elif os.path.isdir(path):  # Verificar si es una carpeta
                        shutil.rmtree(path)  # Eliminar la carpeta y su contenido
                        removed_folders.append(path)

                except Exception as e:
                    print(f"Error al intentar eliminar {path}: {e}")
        else:
            print(f'No existe el directorio: {directory}')

    print(f'Archivos eliminados: {len(removed_files)}')
    print(f'Carpetas eliminadas: {len(removed_folders)}')

# Función para remover páginas seleccionadas
def remove_image(images, selection):

    start_time = time.perf_counter()

    parts = str(images).split("/")
    user_id = parts[-2]

    print('-'*100)
    print('Remove images')
    # Date of request
    current_date = datetime.now()
    current_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Date: {current_date}')
    print(f'User id: {user_id}')

    # Separar por comas (str-lst)
    # 1, 2, 3-5 -> ['1','2','3-5']
    splitted_pages = [page.strip() for page in selection.split(',')]

    # Separar por guiones (rangos de páginas) y convertir a número entero
    # ['1','2','3-5'] - > ['3-5'] (lst-lst)
    pages_ranges = [item for item in splitted_pages if '-' in item]

    # Si encontró rangos
    if len(pages_ranges) > 0:

        # Separar los rangos de páginas
        for page_range in pages_ranges:
            # Eliminar rango de lista splitted_pages (lst-lst)
            # ['1','2','3-5'] - > ['1','2']
            splitted_pages.remove(page_range)

            # Separar rango en inicio y fin (lst-lst)
            # ['3-5'] -> ['3','5']
            split_range = page_range.split('-')

            # Obtener inicio y fin del rango (str-int)
            if split_range[0].strip() != "":
                start = int(split_range[0].strip())
                # ['3','5'] -> 3
            else:
                start = 1

            # Si no hay fin o el fin es mayor que el largo, tomar el largo de la galería (str-int)
            if split_range[-1] == '' or int(split_range[-1].strip()) > int(len(images)):
                # ['3',''] -> 10
                # ['3','150'] -> 10
                end = int(images[-1][1])

            else:
                # ['3','5'] -> 5 (str-int)
                end = int(split_range[-1].strip())

            # Hacer lista de páginas del rango (int - int)
            consecutive_pages_int = list(range(start, end + 1))
            # [3,5] -> [3,4,5]

            # Convertir páginas a string (int - str)
            consecutive_pages_str = [str(page) for page in consecutive_pages_int]
            # [3,4,5] -> ['3','4','5']

            # Extender lista de páginas lst + lst
            splitted_pages.extend(consecutive_pages_str)

        # Ordenar páginas a eliminar str - int
        sorted_pages = sorted([int(page) for page in splitted_pages])

        # Convertir páginas a int
        splitted_pages = [page for page in sorted_pages]

    # Images to remove
    images_to_remove = [(image, page.strip()) for image, page in images if int(page) in splitted_pages]

    # Remove pages
    idx = 0
    for image_path, page in images:
        if (image_path, page) in images_to_remove:
            try:
                if os.path.exists(image_path):
                    idx += 1
                    os.remove(image_path)
                    print(f"Removed {idx} of {len(images_to_remove)}: {image_path}")
                else:
                    print(f"File does not exist: {image_path}")
            except Exception as e:
                print(f"Error removing file {image_path}: {e}")

    end_time = time.perf_counter()
    total_time = end_time - start_time if end_time > start_time else 0
    average_time = total_time / len(images_to_remove) if len(images_to_remove) > 0 else 0


    # Updated Gallery
    updated_gallery = [(image, page.strip()) for image, page in images if int(page) not in splitted_pages]

    # Clean selection
    clean_selection = ''

    print(f'Removed pages: {len(splitted_pages)} of {len(images)} {splitted_pages}')
    print(f'Total images removed: {len(images_to_remove)} Total Time: {total_time:.4f} seg Average Time per Image: {average_time:.4f} seg/img')
    print('-'*100)

    # User message
    gr.Info(f"Se eliminaron {len(images_to_remove)} imágenes.", duration=2)

    return updated_gallery, clean_selection

def get_files_paths(source_dir: str = None):
    if os.path.isdir(source_dir):
        print(f"El directorio '{source_dir}' existe.")
        files_names = os.listdir(source_dir)
        files_paths = [os.path.join(source_dir, file_name) for file_name in files_names]
        return files_paths

    else:
        raise ValueError(f"El directorio '{source_dir}' no existe.")

# FOR RETRIEVAL
def get_knowledge(template_file: str=None):
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

def generate(text, template_file):

    model_name = 'qwen2.5-max:14b'

    # Restart Ollama
    start_ollama_server()

    # Load Model
    run_ollama_model(model_name)

    # Get Knowledge

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
                  },
                  stream=True,
                  )

    text = ''
    for chunk in stream:
        text += chunk['message']['content']
        yield text

def get_template_path(choice):

    choices_catalogue = {
        'antecedente de propiedad': 'template_sections.txt',
        'antecedente de constitución': 'template_constitucion.txt',
    }

    choice_file = choices_catalogue[choice]

    return choice_file

def read_file(file_name):
    with open(file_name, "r") as f:
        return f.read()


########################################################################################
# GRADIO INTERFACE
########################################################################################

# Theme
theme = gr.themes.Default(primary_hue="red")

with (gr.Blocks(css="footer {display: none !important}",
                fill_width=True,
                title="N230-OCR",
                theme=theme,
                #delete_cache=(1,30)
                ) as app):

    logo = gr.Image(
        value="logo.png",
        width=250,
        height=50,
        container=False,
        interactive=False,
        show_fullscreen_button=False,
        show_download_button=False,
        elem_id="logo_cmd"
    )

    ################################################################################################
    # TAB OCR
    ################################################################################################

    # ---------------------------------------------------------------------------------------
    # COMPONENTS
    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    # ROW IMAGES GALLERIES & TEXT
    # ---------------------------------------------------------------------------------------
    with gr.Tab("Leer"):

        with gr.Row():
            # ---------------------------------------------------------------------------------------
            # COLUMN IMAGES GALLERIES
            # ---------------------------------------------------------------------------------------
            with gr.Column(scale=1):
                # -----------------------------
                # Gallery
                # -----------------------------
                '''Galería de Imágenes'''
                gallery = gr.Gallery(
                    label="Galería",
                    elem_id="gallery",
                    object_fit="scale-down",
                    columns=2,
                    rows=1,
                    height="auto",  # "auto" to show the scroll bar / int in pxls but not show scroll bar
                    container=True,
                    show_fullscreen_button=True,    # Muestra ícono para ver en pantalla completa
                    allow_preview=True,             # Permite ver cada imagen
                    preview=False,                  # Inicia en modo preview
                    selected_index=0,               # Índice de imagen seleccionada por defecto
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                )

            # ---------------------------------------------------------------------------------------
            # COLUMN TEXT EXTRACTION
            # ---------------------------------------------------------------------------------------
            with gr.Column(scale=3):
                ocr_result = gr.Textbox(
                    container=True,
                    show_copy_button=True,
                    show_label=True,
                    label="",
                    lines=20,
                    max_lines=20,
                    interactive=True,
                )

        with gr.Row():
            with gr.Column(scale=1):
                '''Botón Cargar Archivo'''
                uploaded_file = gr.UploadButton(
                    variant="primary",
                    label="Cargar",
                    file_count='single',
                    file_types=ACCEPTED_EXTENSIONS,
                    interactive=True,

                )

                '''Caja de Texto de páginas a eliminar'''
                selected_pages = gr.Textbox(
                    lines=1,
                    placeholder='p.e.: 1,2,3,5-10,15',
                    max_length=20,
                    interactive=True,
                    visible=True,
                    show_label=False,
                    label='Eliminar páginas',
                    container=True,
                    submit_btn='Eliminar',
                )

            with gr.Column(scale=1):
                '''Botón para extraer texto de imágenes'''
                ocr_button = gr.Button(
                    scale=1,
                    variant="secondary",
                    value="Leer",
                    elem_id="ocr_button",
                    interactive=False,
                )

                download_button = gr.DownloadButton(
                    scale=1,
                    label="Descargar",
                    variant='secondary',
                    interactive=False,
                    visible=True,
                )

            with gr.Column(scale=2):

                markdown_text = '''<p style="font-size: 12px; text-align: right"><b>IMPORTANTE: REVISE Y CORRIJA ANTES DE USAR EL TEXTO.</b></p>'''
                gr.Markdown(
                    line_breaks=False,
                    value=markdown_text,
                    sanitize_html=False
                )

                queue_status = gr.Markdown(
                    line_breaks=False,
                    value='',
                    sanitize_html=False
                )
    ################################################################################################
    # TAB RETRIEVER
    ################################################################################################
    with (gr.Tab(
            label="Redactar",
            visible=False,
            interactive=False,
            render=False
    ) as redactar_tab):

        # Components
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    inputs=[ocr_result],
                    label='Texto Original',
                    lines=20,
                    max_lines=20,
                    interactive=True,
                    autoscroll=False,
                    show_copy_button=True,
                )

            with gr.Column(scale=2):
                response = gr.Textbox(
                    label='Texto Generado',
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    autoscroll=True,
                    show_copy_button=True,
                )

        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(visible=False)

                task_selection = gr.Dropdown(
                    container=False,
                    # info='Seleccione una tarea',
                    label='Tarea',
                    show_label=False,
                    choices=['antecedente de propiedad','antecedente de constitución']
                )

            with gr.Column(scale=1):
                generate_btn = gr.Button("Redactar", variant="primary")

            with gr.Column(scale=1):
                download_btn = gr.DownloadButton(
                    label="Descargar",
                    variant="secondary"
                )

                markdown_text = '''<p style="font-size: 12px; text-align: right"><b>IMPORTANTE: REVISE Y CORRIJA ANTES DE USAR EL TEXTO.</b></p>'''
                gr.Markdown(
                    line_breaks=False,
                    value=markdown_text,
                    sanitize_html=False
                )


    ################################################################################################
    with gr.Tab("Knowledge", visible=False, interactive=False, render=False):
        with gr.Row():
            with gr.Column(scale=1):
                collections_selector = gr.Dropdown()

            with gr.Column(scale=3):
                glosario_textbox = gr.Textbox()


    ################################################################################################
    # GENERAL EVENTS
    ################################################################################################
    # Save txt file
    # ---------------------------------------------------
    def cleanup_text():
        return gr.update(value='')

    def update_primary():
        return gr.update(variant='primary', interactive=True)

    def update_secondary():
        return gr.update(variant='secondary', interactive=False)

    def update_value(value):
        return gr.update(value=value)

    def interactive_true():
        return gr.update(interactive=True)

    def interactive_false():
        return gr.update(interactive=False)

    # Save txt file
    # ---------------------------------------------------
    def save_text_to_file(uploaded_file, text):

        # Get user id
        parts = str(uploaded_file).split(os.sep)
        user_id = parts[-2]
        # Save restored image
        output_dir = os.path.join('/tmp/gradio', user_id)

        # Clean old txt files in directory
        for item in glob.glob(os.path.join(output_dir, "*.txt")):
            if os.path.isfile(item):
                try:
                    os.unlink(item)
                except:
                    pass

        # Construct file path with timestamp
        # Generate current date and time in yyyymmdd_hhmmss format
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        new_file_path = os.path.join(output_dir, f'ocr_{timestamp}.txt')

        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        return new_file_path

    # ---------------------------------------------------
    # UPLOAD FILE EVENT
    # ---------------------------------------------------

    uploaded_file.upload(
        fn=cleanup_text,                                            # Clean text window
        inputs=None,
        outputs=ocr_result
    ).then(
        fn= queue_manager,                                          # Append to Convert queue
        inputs=[uploaded_file, gr.Number(1,visible=False)],
        outputs=[queue_status]
    ).then(
        fn=convert,                                                 # Convert file to images jpg
        inputs=uploaded_file,
        outputs=[gallery],
        concurrency_limit=1,
        concurrency_id='convert_queue'
    ).then(
        fn=update_secondary,
        inputs=None,
        outputs=[uploaded_file],
    ).then(
        fn=update_primary,
        inputs=None,
        outputs=[ocr_button],
    ).then(
        fn=complete_queue,
        inputs=[uploaded_file,gr.Number(1,visible=False)],
        outputs=None,
    )

    # ---------------------------------------------------
    # REMOVE SELECTED IMAGES EVENT
    # ---------------------------------------------------
    selected_pages.submit(
        fn=remove_image,
        inputs=[gallery, selected_pages],
        outputs=[gallery, selected_pages],
        concurrency_limit=1,
        concurrency_id='remove_queue'
    )


    # ---------------------------------------------------
    # IMAGES RESTORATION AND OCR EVENTS
    # ---------------------------------------------------
    ocr_event = ocr_button.click(
        fn=queue_manager,                                           # Append to Restore queue
        inputs=[uploaded_file, gr.Number(2,visible=False)],
        outputs=queue_status
    ).then(
        fn=update_secondary,                                        # Unable button ocr
        inputs=None,
        outputs=[ocr_button],
    ).then(
        fn=update_secondary,                                        # Unable button download
        inputs=None,
        outputs=[download_button],
    ).then(
        fn=cleanup_text,                                            # Clean text window
        inputs=None,
        outputs=ocr_result
    ).then(
        fn=images_restoration,                                      # Images Restoration
        inputs=[uploaded_file, gallery],
        outputs=[gallery],
        concurrency_limit=1,
        concurrency_id="restore_queue",
    ).then(
        fn=complete_queue,
        inputs=[uploaded_file, gr.Number(2,visible=False)],   # Remove from Restore queue
        outputs=queue_status
    ).then(
        fn=queue_manager,
        inputs=[uploaded_file, gr.Number(3,visible=False)],   # Append to OCR queue
        outputs=queue_status
    ).then(
        fn=qwen_ocr,                                                # Extract text content
        inputs=[uploaded_file, gallery],
        outputs=[ocr_result],
        concurrency_limit=1,
        concurrency_id="ocr_queue",
    ).then(
        fn=save_text_to_file,                                       # Save txt file
        inputs=[uploaded_file, ocr_result],
        outputs=[download_button],
    ).then(
        fn=update_primary,                                          # Change button
        inputs=None,
        outputs=[download_button],
    ).then(
        fn=update_primary,                                          # Change button
        inputs=None,
        outputs=[uploaded_file],
    ).then(
        fn=complete_queue,
        inputs=[uploaded_file, gr.Number(3, visible=False)],  # Remove from OCR Queue
        outputs=None,
    )

    # ---------------------------------------------------
    # TEXT CHANGE EVENT
    # ---------------------------------------------------

    ocr_result.change(
        fn=save_text_to_file,                               # Save txt file
        inputs=[uploaded_file, ocr_result],
        outputs=[download_button],
        concurrency_id="gpu_queue",
    )

    # ---------------------------------------------------
    # DOWNLOAD TEXT FILE EVENT
    # ---------------------------------------------------

    download_button.click(
        fn=update_primary,                              # Change button
        inputs=None,
        outputs=uploaded_file,
    ).then(
        fn=update_secondary,                            # Change button
        inputs=None,
        outputs=download_button,
    )

    ##################################################################
    # TEXT GENERATOR EVENTS
    ##################################################################

    # ---------------------------------------------------
    # TAB SELECTION EVENT
    # ---------------------------------------------------

    redactar_tab.select(
        fn=update_value,
        inputs=[ocr_result],
        outputs=[input_text],
    )

    # ---------------------------------------------------
    # TASK SELECTION CHANGE EVENT
    # ---------------------------------------------------
    task_state = gr.State()
    task_selection.change(
        fn=get_template_path,
        inputs=task_selection,
        outputs=task_state
    )

    # ---------------------------------------------------
    # TEXT GENERATION EVENT
    # ---------------------------------------------------
    generate_btn.click(
        fn=queue_manager,
        inputs=[uploaded_file, gr.Number(2,visible=False)],
        outputs=queue_status
    ).then(
        fn=update_secondary,
        inputs=None,
        outputs=[generate_btn]
    ).then(
        fn=get_template_path,
        inputs=task_selection,
        outputs=task_state
    ).then(fn=generate,
        inputs=[input_text, task_state],
        outputs=[response],
        concurrency_limit=1,
        concurrency_id = "gpu_queue",
    ).then(
        fn=save_text_to_file,
        inputs=[uploaded_file, response],
        outputs=[download_btn]
    ).then(
        fn=interactive_true,
        inputs=None,
        outputs=[response],
    ).then(
        fn=update_primary,
        inputs=None,
        outputs=[download_btn],
    ).then(
        fn=save_text_to_file,
        inputs=[uploaded_file, response],
        outputs=[download_btn]
    )

    # ---------------------------------------------------
    # DOWNLOAD GENERATED TEXT EVENT
    # ---------------------------------------------------
    download_btn.click(
        fn=update_primary,
        inputs=None,
        outputs=generate_btn,
    ).then(
        fn=update_secondary,
        inputs=None,
        outputs=download_btn,
    )



if __name__ == "__main__":

    # Kill Ollama
    kill_ollama_server(wait_time=3)

    # Clean GPU cache
    torch.cuda.empty_cache()

    # Check Memory
    gpus = gpu_checkup()

    for gpu in gpus:
        print(f'Used: {gpu.memoryUsed}GB, Total: {gpu.memoryTotal}GB, Used/Total: {gpu.memoryUtil * 100:.2f}%')

    # Limpiar
    clean_files()

    # Habilitar las colas
    app.queue()

    # Lanzar la app
    app.launch(allowed_paths=["."],
               share=False,
               debug=False,
               server_name="192.168.1.159",     # 192.168.1.159
               server_port=8090,                # 8090
               )