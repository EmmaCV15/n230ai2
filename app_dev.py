import os.path
import glob

import gradio as gr
import torch.cuda
import datetime
import queue

from converter.converter import *
from restormer.restormer import images_restoration
from qwen_ocr.qwen_ocr import qwen_ocr
from config import INPUT_FILES, IMAGES_INPUT, IMAGES_RESTORED
from utils.ollama_utils import *
from utils.gpu_utils import *

convert_queue = queue.Queue()
ocr_queue = queue.Queue()

def complete_conversion(user_id):
    convert_queue.get()

def queue_manager(user_id, step):

    """Maneja la posición del usuario en cada cola"""
    if step == 1:  # Primera cola (Carga de archivo)
        convert_queue_list = list(convert_queue.queue)
        print(len(convert_queue_list))

        if user_id not in convert_queue_list:
            convert_queue.put(user_id)

        # get the updated list
        convert_queue_list = list(convert_queue.queue)

        position = convert_queue_list.index(user_id) + 1 if user_id in convert_queue_list else ocr_queue.qsize()
        total_in_queue = convert_queue.qsize()
        return f'''<p style="font-size: 11px; text-align: right">Conversión: turno {position} de {total_in_queue}</p>'''

    elif step == 2:  # Segunda cola (OCR)

        ocr_queue_list = list(ocr_queue.queue)
        # Si no está en la lista, agregar
        if user_id not in ocr_queue_list:
            ocr_queue.put(user_id)

        # get the updated list
        ocr_queue_list = list(ocr_queue.queue)
        position = ocr_queue_list.index(user_id) + 1 if user_id in ocr_queue_list else ocr_queue.qsize()
        total_in_queue = ocr_queue.qsize()

        return f'''<p style="font-size: 11px; text-align: right">OCR: turno {position} de {total_in_queue}</p>'''


def complete_queue(user_id, step):
    if step == 1:  # Primera cola (Carga de archivo y conversion a imagen)
        convert_queue.get()
    elif step == 2:  # Segunda cola (Correccion Imagen y OCR)
        ocr_queue.get()


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


# Move file to INPUT_FILES
def load_file(path: str = None):
    """
    Copia o mueve un archivo a una carpeta de destino.
    :param path: Ruta del archivo de origen.
    :return: Nueva ruta del archivo.
    """

    if not os.path.isfile(path):
        gr.Warning('Error al cargar el archivo')
        raise FileNotFoundError(f"El archivo '{path}' no existe.")

    base_name = os.path.basename(path)
    new_path = os.path.join(INPUT_FILES, base_name)
    shutil.move(path, new_path)
    return new_path


# Función para remover páginas seleccionadas
def remove_image(images, selection):
    # Separar por comas (str-lst)
    # 1, 2, 3-5 -> ['1','2','3-5']
    splitted_pages = [page.strip() for page in selection.split(',')]

    # Separar por guiones (rangos de páginas) y convertir a número entero
    # ['1','2','3-5'] - > ['3-5'] (lst-lst)
    pages_ranges = [item for item in splitted_pages if '-' in item]

    print('splitted_pages: ',splitted_pages)
    print('pages_ranges: ', pages_ranges)

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
            print('split_range', split_range)

            # Obtener inicio y fin del rango (str-int)
            if split_range[0].strip() != "":
                start = int(split_range[0].strip())
                # ['3','5'] -> 3
            else:
                start = 0

            print('start:', start)

            # Si no hay fin o el fin es mayor que el largo, tomar el largo de la galería (str-int)
            if split_range[-1] == '' or int(split_range[-1].strip()) > int(len(images)):
                # ['3',''] -> 10
                # ['3','150'] -> 10
                end = int(images[-1][1])

            else:
                # ['3','5'] -> 5 (str-int)
                end = int(split_range[-1].strip())

            print('end: ', end)

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
    print('splitted_pages', splitted_pages)

    # Imágenes a eliminar (str)
    images_to_remove = [(image, page.strip()) for image, page in images if int(page) in splitted_pages]
    print('images_to_remove', images_to_remove)

    gr.Info(f"Se eliminaron {len(images_to_remove)} imágenes.", duration=2)

    # Nueva Galería
    new_gallery = [(image, page.strip()) for image, page in images if int(page) not in splitted_pages]
    print('new_gallery: ', new_gallery)

    # Clean selection
    clean_selection = ''

    return new_gallery, clean_selection


def get_files_paths(source_dir: str = None):
    if os.path.isdir(source_dir):
        print(f"El directorio '{source_dir}' existe.")
        files_names = os.listdir(source_dir)
        files_paths = [os.path.join(source_dir, file_name) for file_name in files_names]
        return files_paths

    else:
        raise ValueError(f"El directorio '{source_dir}' no existe.")


########################################################################################
# GRADIO INTERFACE
########################################################################################
with (gr.Blocks(css="footer {display: none !important}", fill_width=True, title="N230-OCR") as app):
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
    with gr.Tab("OCR"):

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
                    height=450,  # "auto" to show the scroll bar / int in pxls but not show scroll bar
                    container=True,
                    show_fullscreen_button=True,  # Muestra ícono para ver en pantalla completa
                    allow_preview=True,  # Permite ver cada imagen
                    preview=False,  # Inicia en modo preview
                    selected_index=0,  # Índice de imagen seleccionada por defecto
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
                input_file = gr.UploadButton(
                    variant="primary",
                    label="Cargar archivo",
                    file_count='single',
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"],
                    interactive=True,
                )

                new_path = gr.Markdown(visible=False)

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
                    value="OCR",
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

                markdown_text = '''<p style="font-size: 11px; text-align: right"><b>IMPORTANTE:</b> ANTES DE USAR EL TEXTO LO DEBE COTEJAR, REVISAR Y CORREGIR.</p>'''
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
    with gr.Tab("Recuperación", visible=False, interactive=False, render=False):

        with gr.Row():
            with gr.Column(scale=1):
                file_path = gr.File(
                    label='Cargar Archivo .txt o .docx',
                    file_count='single',
                    file_types=['.txt', '.docx'],

                )

                tasks = ['Datos Básicos del Instrumento',
                         'Descripción de Inmueble',
                         'Transmisión de Propiedad',
                         'Constitución de Sociedad',
                         'Otorgamiento de Poder']

                task_selector = gr.Dropdown(
                    label="Esquema de Datos",
                    info="Seleccione un esquema de datos",
                    choices=tasks,
                    interactive=True,
                )

                retriever_button = gr.Button('Recuperar')
                download_button_2 = gr.DownloadButton('Descargar')

            with gr.Column(scale=3):
                data_retrieved = gr.Textbox(
                    container=True,
                    show_copy_button=True,
                    show_label=False,
                    label="Datos",
                    lines=21,
                    max_lines=21,
                    interactive=True,
                )

                input_message = gr.Textbox(
                    show_label=False,
                    container=True,
                    placeholder='Pregunte lo que desee sobre el documento.',
                    submit_btn=True,
                )

                gr.Markdown(line_breaks=True,
                            value="""**IMPORTANTE: DEBE COTEJAR, REVISAR Y CORREGIR EL TEXTO ANTES DE USARLO.**"""
                            )
    ################################################################################################
    with gr.Tab("Knowledge", visible=False, interactive=False, render=False):
        with gr.Row():
            with gr.Column(scale=1):
                collections_selector = gr.Dropdown()

            with gr.Column(scale=3):
                glosario_textbox = gr.Textbox()


    ################################################################################################
    # OCR EVENTS
    ################################################################################################
    # .........................................
    # Load and Convert
    # .........................................
    def clean_up_date():
        return gr.update(value='')


    def update_primary():
        return gr.update(variant='primary', interactive=True)


    def update_secondary():
        return gr.update(variant='secondary', interactive=False)


    def update_value(value):
        return gr.update(value=value)


    convertion_event = input_file.upload(
        fn=load_file,
        inputs=[input_file],
        outputs=new_path,
        concurrency_limit=4,
        concurrency_id='load_queue'
    ).then(fn= queue_manager,
           inputs=[input_file, gr.Number(1,visible=False)],
           outputs=[queue_status]
    ).then(
        fn=convert,
        inputs=new_path,
        outputs=[gallery],
        concurrency_id='load_queue'
    ).then(
        fn=update_secondary,
        inputs=None,
        outputs=[input_file],
    ).then(
        fn=update_primary,
        inputs=None,
        outputs=[ocr_button],
    ).then(
        fn=complete_queue(input_file,gr.Number(1,visible=False))
    )

    # Remove selected images
    # .........................................
    selected_pages.submit(
        fn=kill_ollama_server,
        inputs=None,
        outputs=None,
    ).then(
        fn=remove_image,
        inputs=[gallery, selected_pages],
        outputs=[gallery, selected_pages],
    )


    # OCR & Save
    # .........................................
    def save_text_to_file(text):

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Clean files in directory

        for item in glob.glob(os.path.join(PROJECT_PATH, output_dir, "*")):
            try:
                os.unlink(item)
            except IsADirectoryError:
                pass

        # Construct file path with timestamp
        # Generate current date and time in yyyymmdd_hhmmss format
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        new_file_path = os.path.join(output_dir, f'ocr_{timestamp}.txt')

        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        return new_file_path


    def update_download_button(text):
        new_file_path = save_text_to_file(text)
        return gr.update(value=new_file_path)


    ocr_event = ocr_button.click(
        fn=queue_manager,
        inputs=[input_file, gr.Number(2,visible=False)],
        outputs=queue_status
    ).then(
        fn=update_secondary,
        inputs=None,
        outputs=[ocr_button],
        concurrency_limit=1,
        concurrency_id="gpu_queue",
    ).then(
        fn=kill_ollama_server,
        inputs=None,
        outputs=None,
        concurrency_id="gpu_queue",
    ).then(
        fn=images_restoration,  # Image Restoration
        inputs=[gallery],
        outputs=[gallery],
        concurrency_id="gpu_queue",
    ).then(
        fn=qwen_ocr,  # Extract text content
        inputs=[gallery],
        outputs=[ocr_result],
        concurrency_id="gpu_queue",
    ).then(
        fn=save_text_to_file,  # Save File
        inputs=[ocr_result],
        outputs=[download_button],
        concurrency_id="gpu_queue",
    ).then(
        fn=update_primary,
        inputs=None,
        outputs=[download_button],
        concurrency_id="gpu_queue",
    ).then(
        fn=complete_queue,
        inputs=[input_file, gr.Number(2,visible=False)]
    )

    ocr_result.change(
        fn=save_text_to_file,
        inputs=ocr_result,
        outputs=[download_button],
        concurrency_id="gpu_queue",
    )

    download_button.click(
        fn=update_primary,
        inputs=None,
        outputs=input_file,
        concurrency_id="gpu_queue",
    ).then(
        fn=update_secondary,
        inputs=None,
        outputs=download_button,
        concurrency_id="gpu_queue",
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
               server_name="192.168.1.159",
               server_port=8000,
               )