import fitz
import numpy as np
import os
import shutil
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import cupy as cp
import re
import gradio as gr


from utils.gpu_utils import gpu_checkup
from config import PROJECT_PATH, IMAGES_INPUT, INPUT_FILES, ACCEPTED_EXTENSIONS


def validate_file(file_path):

    """Validar si el archivo existe y es de un formato válido."""
    if not os.path.isfile(file_path):
        gr.Warning(f"Archivo no encontrado: {file_path}")
        return False

    _, ext = os.path.splitext(file_path.lower())
    if ext not in ACCEPTED_EXTENSIONS:
        gr.Warning(f"Formato de archivo no soportado: {ext}. Formatos soportados son: {ACCEPTED_EXTENSIONS}")
        return False

    else:
        return True

def calculate_density(image):

    def calculate_density_gpu(image):
        """Calcular la densidad de píxeles oscuros versus claros usando un umbral con CuPy."""

        gray_array = cp.array(image.convert("L"))
        gray_array = cp.asnumpy(gray_array)

        threshold = 128
        dark_pixels = cp.sum(gray_array < threshold)  # Contar píxeles oscuros
        total_pixels = gray_array.size  # Número total de píxeles
        return dark_pixels / total_pixels if total_pixels > 0 else 0

    def calculate_density_cpu(image):
        """Calcular la densidad de píxeles oscuros versus claros usando un umbral con NumPy."""
        gray_array = np.array(image.convert("L"))  # Convertir la imagen a escala de grises
        threshold = 128
        dark_pixels = np.sum(gray_array < threshold)  # Contar píxeles oscuros
        total_pixels = gray_array.size  # Número total de píxeles
        return dark_pixels / total_pixels if total_pixels > 0 else 0

    gpus = gpu_checkup()

    if gpus:
        try:
            return calculate_density_gpu(image)
        except:
            return calculate_density_cpu(image)
    else:
        try:
            calculate_density_cpu(image)
        except Exception as e:
            print(f"Error al calcular densidad: {e}")
            return None

def save_image(i, image=None, input_file:str=None):

    """Procesar y guardar la imagen de forma concurrente."""

    # Convert to gray image
    image = image.convert('L')

    # Calcular la densidad
    density = calculate_density(image)

    # Get Output Directory
    output_directory = os.path.dirname(input_file)

    # Create the output file name
    output_file_name = f"{os.path.splitext(os.path.basename(input_file))[0]}_pag_{str(i + 1).zfill(3)}.jpg"
    output_file = os.path.join(output_directory, output_file_name)
    page = output_file.split('pag_')[-1].split('.')[0]
    # Metadata
    exif_data = {
        'origen': input_file,
        'pagina': page,
        'densidad': str(density),
        'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Save image
    image.save(output_file, 'JPEG')

def convert_pdf2jpeg(file_path):

    """Convertir las páginas de un archivo PDF a imágenes JPEG."""
    try:
        pdf_document = fitz.open(file_path)  # Abrir el documento PDF

        gr.Info(f'Convirtiendo {len(pdf_document)} páginas.', duration=2)

        images = []  # Lista para almacenar las imágenes convertidas
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)  # Cargar cada página
            pix = page.get_pixmap(dpi=300)  # Renderizar la página como imagen
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convertir pixmap a imagen PIL
            images.append(image)  # Agregar la imagen a la lista

        return images  # Retornar todas las imágenes convertidas
    except Exception as e:
        print(f"Error al procesar el archivo PDF: {e}")
        return []

def convert_tiff2jpeg(file_path):
    """Convertir las páginas de una imagen TIFF a imágenes JPEG."""
    try:
        img = Image.open(file_path)  # Abrir la imagen TIFF
        images = []  # Lista para almacenar las imágenes convertidas

        for i in range(img.n_frames):
            img.seek(i)  # Moverse a cada fotograma en el TIFF
            images.append(img.convert("RGB"))  # Convertir cada fotograma a RGB y agregar a la lista

        return images  # Retornar todas las imágenes convertidas
    except Exception as e:
        print(f"Error al procesar la imagen TIFF: {e}")
        return []

def convert_img2jpeg(file_path):
    """Procesar un archivo de imagen (JPEG, PNG, etc.)."""
    try:
        image = Image.open(file_path).convert("RGB")  # Abrir y convertir la imagen a RGB
        return [image]  # Retornar la imagen convertida en una lista
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return []

def process_file(file_path):
    """Procesar un archivo y guardar sus imágenes convertidas con metadatos de densidad."""
    # Validate input file
    jpeg_images = []
    if validate_file(file_path):
        # Get extension
        _, ext = os.path.splitext(file_path.lower())
        # Convert to jpeg depends on ext
        if ext == '.pdf':
            jpeg_images = convert_pdf2jpeg(file_path)
        elif ext in ['.tiff', '.tif']:
            jpeg_images = convert_tiff2jpeg(file_path)
        else:
            jpeg_images = convert_img2jpeg(file_path)

    return jpeg_images


def move_file(file_path:str=None, move=True):
    """
    Copia o mueve un archivo a una carpeta de destino.
    :param file_path: Ruta del archivo de origen.
    :param move: Si es True, mueve el archivo en lugar de copiarlo.
    :return: Nueva ruta del archivo.
    """

    destination_folder = INPUT_FILES

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")

    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)  # Crea la carpeta si no existe

    base_name = os.path.basename(file_path)
    new_path = os.path.join(destination_folder, base_name)

    if move:
        shutil.move(file_path, new_path)
    else:
        shutil.copy2(file_path, new_path)  # copy2 mantiene metadatos

    return new_path

def convert(file_path:str=None):

    print('-'*100)
    print('Convert file to images process:')
    print("-" * 100)
    # Get current date and time
    current_date = datetime.now()
    current_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Date: {current_date}')
    print(f"User id: {file_path}")
    print('Input file validation: ', validate_file(file_path))

    # Output directory
    output_directory = os.path.dirname(file_path)

    """Convierte un archivo y guarda las imágenes resultantes."""
    start_time = time.perf_counter()

    """Procesar un archivo y guardar sus imágenes convertidas con metadatos de densidad."""
    # Validate input file
    validate_file(file_path)

    # Get basename w/o extension and extension
    basename = os.path.basename(file_path)
    file_name, ext = os.path.splitext(basename)

    # Convert to jpeg depends on its extension
    if ext == '.pdf':
        jpeg_images = convert_pdf2jpeg(file_path)       # pdf format
    elif ext in ['.tiff', '.tif']:
        jpeg_images = convert_tiff2jpeg(file_path)      # tiff images format
    else:
        jpeg_images = convert_img2jpeg(file_path)       # All other images formats

    # Save images as jpeg format
    executor = ThreadPoolExecutor(max_workers=32)
    with executor as ex:
        futures = []
        for i, image in enumerate(jpeg_images):
            futures.append(ex.submit(save_image, i, image, file_path))
            for future in futures:
                future.result()

    # Get the converted images from the file
    all_files =  os.listdir(output_directory)
    # Filter files by pattern
    pattern = r'pag_\d{3}\.jpg$'
    images_names = [f for f in all_files if re.search(pattern, f)]
    images_paths = []
    for name in images_names:
        images_paths.append(os.path.join(output_directory, name))

    # get page number from image path
    gallery = []
    for idx, image_path in enumerate(images_paths):

        for device in gpu_checkup():
            print(f'Converted: {idx+1} of {len(images_paths)} Used VRAM: {device.memoryUsed:.0f}GB ({device.memoryUtil * 100:.2f}%) image: {image_path}')

        page_number = int(image_path.split('pag_')[-1].split('.')[0])
        gallery.append((image_path, page_number))

    # Sort gallery by page number
    gallery = sorted(gallery, key=lambda x:x[1])

    # Convert the page number to string
    gallery = [(path, str(page)) for path, page in gallery]

    if images_paths:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        time_per_image = total_time / len(images_paths)
        print(f'Total Converted images: {len(images_paths)} Total Time: {end_time - start_time:.2f} seg Average Time per Image: {time_per_image:.2f} seg/img')
        print('-'*100)

    return gallery


if __name__ == "__main__":

    if gpu_checkup():
        valid_extensions = {".pdf", ".png", ".tif", ".tiff", ".jpeg", ".jpg"}
        valid_files = []

        for root, _, files in os.walk(INPUT_FILES):

            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    valid_files.append(os.path.join(root, file))

        for file_path in valid_files:
            convert(file_path=file_path)
