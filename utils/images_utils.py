import os
from datetime import datetime
import cupy as cp
import numpy as np
from utils.gpu_utils import gpu_checkup
from PIL import Image
from PIL.ExifTags import TAGS

def calculate_density(image):

    def calculate_density_gpu(image):

        """Calcular la densidad de píxeles oscuros versus claros usando un umbral con CuPy."""


        # Convertir pil_image -> grises -> np.array -> cp.array
        pil_image = image.convert('L')
        numpy_image = np.asarray(pil_image)
        gray_array = cp.asarray(numpy_image)

        threshold = 128
        dark_pixels = cp.sum(gray_array < threshold)  # Contar píxeles oscuros
        total_pixels = gray_array.size  # Número total de píxeles
        return dark_pixels / total_pixels if total_pixels > 0 else 0

    def calculate_density_cpu(image):
        """Calcular la densidad de píxeles oscuros versus claros usando un umbral con NumPy."""

        # Convertir pil_image -> grises -> np.array -> cp.array
        pil_image = image.convert('L')
        numpy_image = np.asarray(pil_image)
        gray_array = numpy_image

        threshold = 128
        dark_pixels = np.sum(gray_array < threshold)  # Contar píxeles oscuros
        total_pixels = gray_array.size  # Número total de píxeles
        return dark_pixels / total_pixels if total_pixels > 0 else 0

    if gpu_checkup():
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

def save_image(image=None, file_name:str=None, output_dir:str=None):

    """Procesar y guardar la imagen procesada."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio {output_dir} creado!")

    # Obtener densidad de la imagen de salida
    density = calculate_density(image)

    # Ruta del archivo de salida
    output_file_name = os.path.basename(file_name)
    output_file = os.path.join(output_dir, output_file_name)  # Definir el nombre del archivo de salida

    # Metadata
    exif_data = {
        'source_file': file_name,
        'image_page': output_file.split('pag_')[-1].split('.')[0],
        'pixels_density': f'{density:.4f}',
        'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Usar el método convencional sincrónico de PIL
    image.save(output_file, 'JPEG')

    # print(f"Imagen guardada: {exif_data}")
