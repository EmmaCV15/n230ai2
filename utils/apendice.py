from pdf2image import convert_from_path
import shutil
import os
import numpy as np
import cv2
import pyzbar
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
import chardet
import matplotlib.pyplot as plt
import zipfile
import datetime
import gradio as gr


def read_qr(image, fonts_path):
    checkpoints = False

    # Convert image to numpy array
    np_image = np.array(image)

    # Convert image to gray scale
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    canny = cv2.Canny(gray, 50, 150)
    canny = cv2.dilate(canny, None, iterations=2)
    canny = cv2.erode(canny, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Verify if the approx is rectangular with specific area
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h

            if area > 10000:
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    # Crop and decode the ROI
                    roi = np_image[y:y + h, x:x + w]
                    decoded_list = decode(roi)

                    # Process decoded QR codes
                    for qr_code in decoded_list:
                        # Detect the encoding with `chardet`
                        detected_encoding = chardet.detect(qr_code.data)['encoding']
                        data = qr_code.data.decode(detected_encoding)
                        print(f'data: {data}')

                        # Dictionary to replace characters
                        replacements = {
                            '當': 'ác',
                            '塶': 'ác',
                            'ﾁ':  'Á',
                            '炓': 'Á',
                            '磴': 'ÁE',
                            '簇': 'ÁL',
                            '篾': 'ÁN',
                            '糠': 'ÁR',
                            '糜': 'ÁS',
                            '糟': 'ÁV',
                            '績': 'ÁZ',

                            '馘': 'éd',
                            '嶮': 'éd',
                            'ﾉ':  'É',
                            '冇': 'ÉN',
                            '仉': 'ÉS',
                            '仂': 'ÉR',

                            'ﾍ':  'Í',
                            '泝': 'ÍA',
                            '泩': 'ÍN',
                            '泀': 'ÍG',
                            '泑': 'ÍO',
                            '炅': 'ÍR',

                            '': 'ón',
                            '鏮': 'ón',
                            'ﾓ':  'Ó',
                            '粈': 'ÓM',
                            '粍': 'ÓN',
                            '紞': 'ÓP',

                            'ﾚ':  'Ú',
                            '硢': 'ÚÑ',
                            '湒': 'ÚS',

                            'ﾑ':  'Ñ',
                            '唎': 'ÑE',
                            '埆': 'ÑO',


                        }

                        # Replace characters
                        for key, value in replacements.items():
                            data = data.replace(key, value)

                        # Clean decoded text
                        clean_text = '\n'.join(line.strip() for line in data.split('\n') if line.strip())

                        # Create white image
                        width, height = 600, 210
                        white_image = Image.new('RGB', (width, height), (255, 255, 255))
                        draw = ImageDraw.Draw(white_image)

                        # Verificar si el directorio de fuentes existe
                        if not os.path.exists(fonts_path):
                            print(f"El directorio de fuentes no existe: {fonts_path}")

                        font_path = os.path.join(fonts_path, 'ARIAL.TTF')
                        font_path = os.path.normpath(font_path)

                        # Cargar la fuente con manejo de excepciones
                        try:
                            font = ImageFont.truetype(font_path, 12)

                        except IOError:
                            print(f"No se pudo cargar la fuente en: {font_path}")
                            font = None  # O asignar una fuente predeterminada

                        # Draw the text line by line
                        y_position = 10
                        for line in clean_text.split('\n'):
                            # Identify the name of the acreditado
                            if 'acreditado' in line.lower():
                                nombre_acreditado = line.split(':')[1].strip()
                                nombre_acreditado = nombre_acreditado.split('/')[0].strip()

                            draw.text((10, y_position), line, fill=(0, 0, 0), font=font)
                            y_position += 20  # Adjust the inter-linear space

                        if checkpoints:
                            plt.imshow(white_image, cmap='gray')
                            plt.title('image qr 1')
                            plt.axis('on')
                            plt.show()

                        # Convert the PIL image to numpy array
                        white_image = np.array(white_image)

                        return (white_image, nombre_acreditado)

    print("No se detectó ningún código QR.")
    return None


def convertir_a_imagenes(file_path=None):
    try:
        # Verificamos que el archivo tenga la extensión PDF
        if file_path.lower().endswith(".pdf"):
            # Convertimos el PDF en imágenes
            images = convert_from_path(file_path)
            return images

    except Exception as e:
        print(f"Error al convertir el PDF: {e}")
        return None


def resize_certificate(file_path):
    # Convert PDF file to images
    images = convert_from_path(file_path)
    resized_images = []

    for image in images:
        # Convert the image to numpy array
        image = np.array(image)
        (h, w) = image.shape[:2]
        new_h = int(h * 1.2)
        ratio = new_h / h
        dim = (int(w * ratio), new_h)

        # Resize image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)

    if len(resized_images) == 0:
        return None
    else:
        return resized_images[0]


def resize_qr_image(qr_image, new_w):
    # Verify the imagen
    if qr_image is None:
        return None

    if not isinstance(qr_image, np.ndarray):
        return None

    # Get originals dimensions of the np image
    (h, w) = qr_image.shape[:2]

    # Calculate the new height and the aspect ratio
    ratio = new_w / w
    dim = (new_w, int(h * ratio))

    # Resize the image
    resized_qr_image = cv2.resize(qr_image, dim, interpolation=cv2.INTER_AREA)

    return resized_qr_image


def create_pages(
                resized_certificate,
                resized_qr_image,
                images_path,
                fonts_path
                ):

    # Dimensions of legal sheet in pixels (300 DPI)
    width_px = int(8.5 * 300)  # Width in pixels
    height_px = int(14 * 300)  # Height in pixels

    # --------------------------------------------------------------------------
    # First page
    # --------------------------------------------------------------------------
    # Create a white image
    white_image_1 = Image.new('RGB', (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(white_image_1)

    # Load the font
    font_path = os.path.join(fonts_path, 'ARIALBD.TTF')
    font_path = os.path.normpath(font_path)

    try:
        font = ImageFont.truetype(font_path, 100)

    except IOError:
        print(f"No se pudo cargar la fuente en: {font_path}")
        font = None  # O asignar una fuente predeterminada

    # Get dimension of the image to paste
    resized_certificate = Image.fromarray(resized_certificate, 'RGB')
    (w_paste, h_paste) = resized_certificate.size

    # Calculate the coordinates top-left to center the image
    x_offset = (width_px - w_paste) // 2
    y_offset = (height_px - h_paste) // 2

    # Write text in the white image
    draw.text((x_offset + w_paste - 300, 500), '“A”', fill=(0, 0, 0), font=font)

    # Paste the resized certificate image centered into the white image
    white_image_1.paste(resized_certificate, (x_offset, y_offset))

    # Convert to gray scale image and save it
    white_image_1 = white_image_1.convert('L')

    # Save
    white_image_1.save(f'{images_path}/page_1.png')

    # --------------------------------------------------------------------------
    # Second page
    # --------------------------------------------------------------------------
    # Create a white image
    white_image_2 = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

    # Get the dimension of the resized_qr_image
    (h_paste, w_paste) = resized_qr_image.shape[:2]

    # Calculate the coordinates top-left to center the image
    x_offset = (width_px - w_paste) // 2
    y_offset = 400  # Fixed Y position

    # Paste the QR image
    white_image_2[y_offset:y_offset + h_paste, x_offset:x_offset + w_paste] = resized_qr_image

    # Draw hyphens
    y_position_hyphens = h_paste + y_offset
    num_lines = 50

    for i in range(num_lines):
        hyphens = "-" * 100
        initial_position = (300, y_position_hyphens)
        color = (0, 0, 0)
        thickness = 1

        # Use putText to draw the hyphens
        cv2.putText(white_image_2, hyphens, initial_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)

        # Increase the Y position for next line
        y_position_hyphens += 60

    # Save the page
    cv2.imwrite(f'{images_path}/page_2.png', white_image_2)
def zip_and_download():
    # Define paths
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    apendices_path = os.path.join(project_path, 'apendices')
    zips_path = os.path.join(project_path, 'zips')

    # Genera el nombre del archivo ZIP
    zip_file_name = f"scotiabank_apendices_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_file_path = os.path.join(zips_path, zip_file_name)

    # Crea el archivo ZIP
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for filename in os.listdir(apendices_path):
            if filename.endswith('.pdf'):   # pdf files only
                file_path = os.path.join(apendices_path, filename)
                # Especifica la ruta dentro del ZIP relativa a "apendices_path" en lugar de "project_path"
                zip_file.write(file_path, os.path.basename(file_path))

    # Devuelve la ruta del archivo ZIP creado
    return zip_file_path

def main_apendice(files_paths=None,
                  certificates_path=None,
                  fonts_path=None,
                  images_path=None,
                  apendices_path=None,
                  progress=gr.Progress()
                  ):
    # Check Points
    checkpoints = False

    # Remove all old PDF Files in certificates folder
    for file_name in os.listdir(certificates_path):
        # Build the full file path
        file_path = os.path.join(certificates_path, file_name)
        # Check if it is a file
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save new PDF Files in certificates folder
    for file_path in files_paths:

        # Get the base name of the PDF File
        base_name = os.path.basename(file_path).lower()

        # Define the destination path to copy the PDF files
        destination_path = os.path.join(certificates_path, base_name)
        # Normalize the destination path
        destination_path = os.path.normpath(destination_path)
        # Copy the PDF file to the destination path
        shutil.copy(file_path, destination_path)

    # List all files in the certificate folder
    files_in_certificates = os.listdir(certificates_path)

    # Filter for PDF files
    pdf_files_paths = [os.path.join(certificates_path, file) for file in files_in_certificates if file.endswith('.pdf')]
    normalized_paths = [os.path.normpath(path) for path in pdf_files_paths]

    # For each PDF file
    for file_path in progress.tqdm(normalized_paths):

        # Convert pdf to images
        images = convertir_a_imagenes(file_path)

        # Read first image
        try:
            qr_image, file_name = read_qr(images[0], fonts_path)
        except Exception as e:
            print(f'Error: {e}')
            continue

        if checkpoints:
            plt.imshow(qr_image, cmap='gray')
            plt.title('qr_image')
            plt.axis('on')
            plt.show()

        # Resize certificate
        resized_certificate = resize_certificate(file_path)

        if checkpoints:
            plt.imshow(resized_certificate, cmap='gray')
            plt.title('resized_certificate')
            plt.axis('on')
            plt.show()

        # Resize qr image
        if isinstance(resized_certificate, np.ndarray):
            (h, w) = resized_certificate.shape[:2]
        else:
            continue
        resized_qr_image = resize_qr_image(qr_image, w)

        if checkpoints:
            plt.imshow(resized_qr_image, cmap='gray')
            plt.title('resized_qr_image')
            plt.axis('on')
            plt.show()

        # Create pages
        create_pages(resized_certificate, resized_qr_image, images_path, fonts_path)

        # Create PDF file with the images

        # Open page 1 (resized certificate) and page 2 (resized OCR data)
        page_1 = Image.open(os.path.normpath(os.path.join(images_path, 'page_1.png')))
        page_2 = Image.open(os.path.normpath(os.path.join(images_path, 'page_2.png')))

        file_name = f'QR {file_name}.pdf'

        # Save PDF file in apendices folder
        page_1.save(os.path.join(apendices_path, file_name), save_all=True, append_images=[page_2])

        # Remove all temporary images
        for file_name in os.listdir(images_path):
            # Build the full file path
            file_path = os.path.join(images_path, file_name)
            # Check if it is a file
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Return the apendices list and create the zip file
    apendices = os.listdir(apendices_path)

    if len(apendices) > 0:

        zip_path = zip_and_download()
        apendices_list = '\n'.join([apendice for apendice in apendices])

        # Elimina los archivos .pdf de la carpeta apendices/
        for filename in os.listdir(apendices_path):
            if filename.endswith('.pdf'):
                os.remove(os.path.join(apendices_path, filename))

        # Elimina los archivos .pdf de la carpeta certificates/
        for filename in os.listdir(certificates_path):
            if filename.endswith('.pdf'):
                os.remove(os.path.join(certificates_path, filename))

        return (apendices_list, zip_path)
    else:
        return 'No hay archivos que procesar'
