import os.path
import shutil

# DEFINE WORK DIRECTORIES
WORKING_DIRECTORIES = {
'INPUT_FILES' : "files",
'IMAGES_INPUT' : "images_input" ,
'IMAGES_RESTORED' : "images_restored",
'TMP_FILES' : 'tmp_files',
'TMP_IMAGES' : 'tmp_images',
'TMP_BOXES' : 'tmp_boxes',
'OCR_TEXT' : 'ocr_text',
}

def gr_clean_files():

    """ Eliminar todos los archivos temporales de los siguientes directorios:"""

    # Make a list of working directories
    directories = [os.path.join('..', directory) for directory in list(WORKING_DIRECTORIES.values())]

    # For each directory remove all files.
    for directory in directories:

        if os.path.exists(directory) and os.path.isdir(directory):

            for filename in os.listdir(directory):

                file_path = os.path.join(directory, filename)

                try:
                    if os.path.isfile(file_path):  # Verificar si es un archivo
                        os.unlink(file_path)  # Eliminar el archivo

                    elif os.path.isdir(file_path):  # Verificar si es una carpeta
                        shutil.rmtree(file_path)  # Eliminar la carpeta y su contenido

                except Exception as e:
                    print(f"Error al intentar eliminar {file_path}: {e}")
        else:
            print(f'No existe el directorio: {directory}')
            continue



if __name__ == '__main__':
    gr_clean_files()
