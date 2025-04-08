import os
import re
import shutil
import gradio as gr
from utils.ditto import main_ditto
from utils.apendice import main_apendice
from utils.testimonio import main_testimonio


# AUXILIAR FUNCTIONS
# Configurations
def validate_date(date):

    patterns = [
                r"^(0[1-9]|[12][0-9]|3[01])(\/|-)(0[13578]|1[0,2])(\/|-)(19|20)\d{2}$", # hasta 31 dias
                r"^(0[1-9]|[12][0-9]|30)(\/|-)(0[469]|11)(\/|-)(19|20)\d{2}$",          # hasta 30 dias
                r"^(0[1-9]|[12][0-9])(\/|-)(02)(\/|-)(19|20)\d{2}$"                     # hasta 29 dias
               ]

    patterns = '|'.join(patterns)

    string_date = str(date)

    if re.match(patterns, string_date):
        return string_date
    else:
        gr.Warning("Formato de fecha inválido. Usa dd/mm/aaaa y verifica el día y mes.")
        return string_date

def get_dittos_models():
    return os.listdir('models_dittos/')
def get_closing_models():
    return os.listdir('models_closing/')
def get_pies_models():
    return os.listdir('models_pies/')

def update_ditto_model_selector():
    return gr.update(choices=get_dittos_models(), value='Modelo Escritura Cancelaciones Scotiabank.docx')
def update_closing_model_selector():
    return gr.update(choices=get_closing_models(), value='Modelo Cierre Cancelaciones Scotiabank.docx')
def update_pie_model_selector():
    return gr.update(choices=get_pies_models(), value='Modelo Pie Cancelaciones Scotiabank.docx')

def load_ditto_models(files_paths):
    """Carga nuevos modelos desde archivos seleccionados."""
    if files_paths:
        for file_path in files_paths:
            filename = os.path.basename(file_path)  # Obtiene el nombre base del archivo
            shutil.copy(file_path.name, os.path.join('models_dittos/', filename))
    return get_dittos_models()
def load_closing_models(files_paths):
    """Carga nuevos modelos desde archivos seleccionados."""
    if files_paths:
        for file_path in files_paths:
            filename = os.path.basename(file_path)  # Obtiene el nombre base del archivo
            shutil.copy(file_path.name, os.path.join('models_closing/', filename))
    return get_closing_models()
def load_pie_models(files_paths):
    """Carga nuevos modelos desde archivos seleccionados."""
    if files_paths:
        for file_path in files_paths:
            filename = os.path.basename(file_path)  # Obtiene el nombre base del archivo
            shutil.copy(file_path.name, os.path.join('models_pies/', filename))
    return get_pies_models()

def remove_selected_ditto(model):
    try:
        if model == 'Modelo Escritura Cancelaciones Scotiabank.docx':
            return gr.Info(f'El modelo {model} no se puede eliminar')
        else:
            os.remove(os.path.join('models_dittos/', model))  # Elimina el modelo seleccionado
            gr.Info(f'Se ha eliminado el modelo {model}')
            return get_dittos_models()  # Retorna la lista actualizada de modelos
    except Exception as e:
        return str(e)
def remove_selected_closing(model):
    try:
        if model == 'Modelo Cierre Cancelaciones Scotiabank.docx':
            return gr.Info(f'El modelo {model} no se puede eliminar')
        else:
            os.remove(os.path.join('models_closing/', model))  # Elimina el modelo seleccionado
            gr.Info(f'Se ha eliminado el modelo {model}')
            return get_closing_models()  # Retorna la lista actualizada de modelos
    except Exception as e:
        return str(e)
def remove_selected_pie(model):
    try:
        if model == 'Modelo Pie Cancelaciones Scotiabank.docx':
            return gr.Info(f'El modelo {model} no se puede eliminar')
        else:
            os.remove(os.path.join('models_pies/', model))  # Elimina el modelo seleccionado
            gr.Info(f'Se ha eliminado el modelo {model}')
            return get_pies_models()  # Retorna la lista actualizada de modelos
    except Exception as e:
        return str(e)


def save_dates_config(signature_date=None, testimonio_date=None):

    config_file = 'config'

    configurations = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            for line in file:
                # Remove white spaces
                line = line.strip()
                # Remove empty lines
                if line:
                    # Preserve comments (#)
                    if line.startswith("#"):
                        # Save comments (#)
                        configurations[line.strip()] = ""
                    # Other than comments
                    else:
                        # lines with = is an attribute
                        if "=" in line:
                            # Attributes without value
                            if line.endswith("="):
                                configurations[line[:-1].strip()] = ''  # Guardar clave sin valor
                            else:
                                # Attributes with value, split key value
                                key, value = line.split("=")
                                # Save key value (remove spaces)
                                configurations[key.strip()] = value.strip()
                        else:
                            # Omit lines without =
                            continue

    configurations["FECHA_FIRMA"] = signature_date
    configurations["FECHA_TESTIMONIO"] = testimonio_date

    # Rewrite the config file
    with open(config_file, "w") as file:
        for key, value in configurations.items():
            # Write comments
            if key.startswith("#"):
                file.write(f"{key}\n")
            # Write Attributes add =
            else:
                file.write(f"{key}={value}\n")
    return

def save_ditto_config(
        btn_model_ditto_selector=None,
        top_margin_ditto=None,
        bottom_margin_ditto=None,
        left_margin_ditto=None,
        right_margin_ditto=None,
        binding_margin_ditto=None,
        position_tab_ditto=None,

):
    # Nombre del archivo de configuraciones
    config_file = "config"

    # Verificar si el archivo ya existe para hacer un backup
    if os.path.exists(config_file):
        # Hacer una copia de seguridad
        backup_archivo = "backup_config"
        shutil.copy(config_file, backup_archivo)

    # Leer el archivo de configuraciones existente
    configurations = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            for line in file:
                # Quitar espacios en blanco a los extremos
                line = line.strip()
                # Verificar que la línea no esté vacía
                if line:
                    # Si es un comentario, conservarlo sin cambiarlo
                    if line.startswith("#"):
                        configurations[line.strip()] = ""  # Guardar el comentario
                    # Sino es comentario
                    else:
                        # Si la linea tiene signo de = es un atributo
                        if "=" in line:
                            # Si no tiene un valor, solo conservar la clave
                            if line.endswith("="):
                                configurations[line[:-1].strip()] = ''  # Guardar clave sin valor
                            else:
                                # Separarlo en calve, valor
                                key, value = line.split("=")
                                configurations[key.strip()] = value.strip()  # Guardar clave y valor sin espacios
                        else:
                            # Si no tiene un '=', la línea no es válida, omitir
                            continue

    # Actualizar las configuraciones con los nuevos valores
    configurations["MODELO_ESCRITURA"] = str(btn_model_ditto_selector)
    configurations["MARGEN_SUPERIOR_DITTO"] = str(top_margin_ditto)
    configurations["MARGEN_INFERIOR_DITTO"] = str(bottom_margin_ditto)
    configurations["MARGEN_IZQUIERDO_DITTO"] = str(left_margin_ditto)
    configurations["MARGEN_DERECHO_DITTO"] = str(right_margin_ditto)
    configurations["MARGEN_ENCUADERNACION_DITTO"] = str(binding_margin_ditto)
    configurations["TAB_POSICION_DITTO"] = str(position_tab_ditto)

    # Escribir todas las configuraciones de vuelta al archivo
    with open(config_file, "w") as file:
        for key, value in configurations.items():
            # Escribir comentarios tal cual
            if key.startswith("#"):
                file.write(f"{key}\n")
            else:
                file.write(f"{key}={value}\n")
    return gr.Info("Configuración Actualizada")



def save_testimonio_config(
        closing_model_selector=None,
        pie_model_selector=None,
        top_margin_testimonio=None,
        bottom_margin_testimonio=None,
        left_margin_testimonio=None,
        right_margin_testimonio=None,
        binding_margin_testimonio=None,
        position_tab_testimonio=None,
):
    # Nombre del archivo de configuraciones
    config_file = "config"

    # Verificar si el archivo ya existe para hacer un backup
    if os.path.exists(config_file):
        # Hacer una copia de seguridad
        backup_archivo = "backup_config"
        shutil.copy(config_file, backup_archivo)

    # Leer el archivo de configuraciones existente
    configurations = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            for line in file:
                # Quitar espacios en blanco a los extremos
                line = line.strip()
                # Verificar que la línea no esté vacía
                if line:
                    # Si es un comentario, conservarlo sin cambiarlo
                    if line.startswith("#"):
                        configurations[line.strip()] = ""  # Guardar el comentario
                    # Sino es comentario
                    else:
                        # Si la linea tiene signo de = es un atributo
                        if "=" in line:
                            # Si no tiene un valor, solo conservar la clave
                            if line.endswith("="):
                                configurations[line[:-1].strip()] = ''  # Guardar clave sin valor
                            else:
                                # Separarlo en calve, valor
                                key, value = line.split("=")
                                configurations[key.strip()] = value.strip()  # Guardar clave y valor sin espacios
                        else:
                            # Si no tiene un '=', la línea no es válida, omitir
                            continue

    # Actualizar las configuraciones con los nuevos valores
    configurations["MODELO_CIERRE"] = str(closing_model_selector)
    configurations["MODELO_PIE"] = str(pie_model_selector)
    configurations["MARGEN_SUPERIOR_TESTIMONIO"] = str(top_margin_testimonio)
    configurations["MARGEN_INFERIOR_TESTIMONIO"] = str(bottom_margin_testimonio)
    configurations["MARGEN_IZQUIERDO_TESTIMONIO"] = str(left_margin_testimonio)
    configurations["MARGEN_DERECHO_TESTIMONIO"] = str(right_margin_testimonio)
    configurations["MARGEN_ENCUADERNACION_TESTIMONIO"] = str(binding_margin_testimonio)
    configurations["TAB_POSICION_TESTIMONIO"] = str(position_tab_testimonio)

    # Escribir todas las configuraciones de vuelta al archivo
    with open(config_file, "w") as file:
        for key, value in configurations.items():
            # Escribir comentarios tal cual
            if key.startswith("#"):
                file.write(f"{key}\n")
            else:
                file.write(f"{key}={value}\n")
    return gr.Info("Configuración Actualizada")

# -------------------------------------------
# RUTAS DE FOLDERS Y ARCHIVOS
# -------------------------------------------
def read_configuration_file(file_path):
    configurations = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    configurations[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"El archivo {file_path} no fue encontrado.")
    return configurations

def get_paths():
    paths={}

    # Read the configuration file
    configurations = read_configuration_file('config')

    # MODELS
    # ------------------------------------------------

    # Ditto Model
    # ------------------------------------------------
    ditto_model_name = configurations['MODELO_ESCRITURA']

    # Ditto Model Path
    ditto_model_path = os.path.join(os.getcwd(), 'models_dittos', f'{ditto_model_name}')
    ditto_model_path = os.path.normpath(ditto_model_path)
    # ------------------------------------------------

    # Closing Model
    # ------------------------------------------------
    closing_model_name = configurations['MODELO_CIERRE']

    # Ditto Model Path
    closing_model_path = os.path.join(os.getcwd(), 'models_closing', f'{closing_model_name}')
    closing_model_path = os.path.normpath(closing_model_path)

    # Pie Model
    # ------------------------------------------------
    pie_model_name = configurations['MODELO_PIE']

    # Pie Model Path
    pie_model_path = os.path.join(os.getcwd(), 'models_pies', f'{pie_model_name}')
    pie_model_path = os.path.normpath(pie_model_path)



    # FOLDERS
    # ------------------------------------------------
    # Dittos Folder
    dittos_models_folder = os.path.join(os.getcwd(), 'models_dittos')
    os.makedirs(dittos_models_folder, exist_ok=True)
    dittos_models_folder = os.path.normpath(dittos_models_folder)

    # Closing Folder
    closing_models_folder = os.path.join(os.getcwd(), 'models_closing')
    os.makedirs(closing_models_folder, exist_ok=True)
    closing_models_folder = os.path.normpath(closing_models_folder)

    # Pies Folder
    pie_models_folder = os.path.join(os.getcwd(), 'models_pies')
    os.makedirs(pie_models_folder, exist_ok=True)
    pie_models_folder = os.path.normpath(pie_models_folder)

    # Certificates Folder
    certificates_folder = os.path.join(os.getcwd(), 'certificates')
    os.makedirs(certificates_folder, exist_ok=True)
    certificates_folder = os.path.normpath(certificates_folder)

    # Dittos Folder
    dittos_folder = os.path.join(os.getcwd(), 'dittos')
    os.makedirs(dittos_folder, exist_ok=True)
    dittos_folder = os.path.normpath(dittos_folder)

    # Images Folder
    images_folder = os.path.join(os.getcwd(), 'images')
    os.makedirs(images_folder, exist_ok=True)
    images_folder = os.path.normpath(images_folder)

    # fonts
    fonts_folder = os.path.join(os.getcwd(), 'fonts')
    os.makedirs(fonts_folder, exist_ok=True)
    fonts_folder = os.path.normpath(fonts_folder)

    # apendices
    apendices_folder = os.path.join(os.getcwd(), 'apendices')
    os.makedirs(apendices_folder, exist_ok=True)
    apendices_folder = os.path.normpath(apendices_folder)

    # Data Folder
    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)
    data_folder = os.path.normpath(data_folder)

    # Testimonios Folder
    testimonios_folder = os.path.join(os.getcwd(), 'testimonios')
    os.makedirs(testimonios_folder, exist_ok=True)
    testimonios_folder = os.path.normpath(testimonios_folder)



    # Create a dictionary
    paths['ditto_model_name'] = ditto_model_name
    paths['ditto_model_path'] = ditto_model_path
    paths['closing_model_name'] = closing_model_name
    paths['closing_model_path'] = closing_model_path
    paths['pie_model_name'] = pie_model_name
    paths['pie_model_path'] = pie_model_path

    paths['dittos_models_folder'] = dittos_models_folder
    paths['closing_models_folder'] = closing_models_folder
    paths['pie_models_folder'] = pie_models_folder

    paths['certificates_folder'] = certificates_folder
    paths['dittos_folder'] = dittos_folder
    paths['images_folder'] = images_folder
    paths['fonts_folder'] = fonts_folder
    paths['apendices_folder'] = apendices_folder
    paths['data_folder'] = data_folder
    paths['testimonios_folder'] = testimonios_folder

    return paths

paths = get_paths()

# -------------------------------------------


# -------------------------------------------
# Gradio Interface
# -------------------------------------------
with gr.Blocks(css="footer{display:none !important}", title="N230-Scotiabank") as app:

    configurations = read_configuration_file('config')

    with gr.Row():
        with gr.Column():
            # HEADER
            logo = "logo.png"
            logo_cmd = gr.Image(logo,
                                width=250,
                                container=False,
                                interactive=False,
                                show_fullscreen_button=False,
                                show_download_button=False,

                                )
        with gr.Column():
            pass
        with gr.Column():
            pass
    with gr.Row():
        subtitle = f"""<h3>Cancelaciones de Hipoteca Scotiabank</h3>"""
        gr.HTML(subtitle, elem_id="subtitle")
    # ---------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------
    # TAB DITTOS
    # Generate Dittos from MS Excel File
    # ---------------------------------------------------------------------------------------
    with gr.Tab("Dittos") as dittos_tab:

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Modelo configurado")

                ditto_model_selected = gr.Text(
                    value=get_paths()['ditto_model_name'],
                    lines=1,
                    container=False,
                    interactive=False,
                )

                gr.Markdown("Cargue la base de datos de MS Excel(xlsx)")

                # Cargar base de datos xlsx
                file_input = gr.File(label="",
                                      scale=1,
                                      file_count="single",
                                      type="filepath",
                                      height=270,
                                      )

                process_button = gr.Button(elem_id="process_button", value="Procesar", interactive=False)

                zip_file_path = gr.DownloadButton(elem_id="download_button_dittos", label="Descargar",
                                                  interactive=False)

                clean_button = gr.Button(elem_id="clean_button", value="Limpiar", interactive=True)

            with gr.Column(scale=8):
                gr.Markdown("Resultados")
                output_text = gr.Textbox(label="",
                                         lines=20,
                                         max_lines=20,
                                         autoscroll=False,
                                         )

        # With Change in File Input...
        def change_process_button(input_file):
            return gr.update(interactive=input_file is not None)

        file_input.change(fn=change_process_button, inputs=file_input, outputs=process_button)

        # Process
        # Get ditto margins configurations
        ditto_margins = [configurations['MARGEN_SUPERIOR_DITTO'],
                         configurations['MARGEN_INFERIOR_DITTO'],
                         configurations['MARGEN_IZQUIERDO_DITTO'],
                         configurations['MARGEN_DERECHO_DITTO'],
                         configurations['MARGEN_ENCUADERNACION_DITTO'],
                         configurations['TAB_POSICION_DITTO'],
                         ]

        # Call `main_ditto` utils function
        process_button.click(
            # Unable process button
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=process_button
        ).then(
            # Call main_ditto utils function
            fn=lambda x: main_ditto(excel_path=x,
                                    data_path=paths['data_folder'],         # data folder to get data file
                                    model_path=paths['ditto_model_path'],   # ditto model path to use
                                    dittos_path=paths['dittos_folder'],     # dittos folder to save generated dittos
                                    ditto_margins=ditto_margins,            # ditto margins configuration
                                    progress=gr.Progress()
                                    ),
            inputs=file_input,                                              # input database file
            outputs=[output_text, zip_file_path]                            # list of generated dittos names, zip path to download
        ).then(
            # Enable Download button when process finish
            fn=lambda: gr.update(elem_id="download_button_dittos", interactive=True),
            inputs=None,
            outputs=zip_file_path
        )

        # Clean textbox and file input components
        clean_button.click(
            fn=lambda: ("", None),
            inputs=None,
            outputs=[output_text, file_input]
        ).then(
            # Enable process button when cleaning finish
            fn=lambda: gr.update(elem_id="process_button", interactive=False),
            inputs=None,
            outputs=process_button
        ).then(
            # Unable download button when cleaning finish
            fn=lambda: gr.update(elem_id="download_button_dittos", interactive=False),
            inputs=None,
            outputs=zip_file_path
        )
    # ---------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------
    # TAB APENDICES
    # Generate Apéndice Letter "A" with certificates PDF files
    # ---------------------------------------------------------------------------------------
    with gr.Tab("Apéndices") as apendices_tab:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Cargue los certificados no adeudo (pdf)")
                file_input = gr.Files(label="", scale=1, file_count="multiple", type="filepath", height=270, )
                # Buttons
                process_button = gr.Button("Procesar")
                zip_file_path = gr.DownloadButton(elem_id="download_button_apendices", label="Descargar",
                                                  interactive=False)
                clean_button = gr.Button("Limpiar")

            with gr.Column(scale=8):
                gr.Markdown("Resultados")
                output_text = gr.Textbox(label="",
                                         lines=20,
                                         max_lines=20,
                                         autoscroll=False,
                                         interactive=False,
                                         )

        # --------------------------------------------------------------------
        # Función para procesar los archivos PDF
        process_button.click(
            # Inhabilita el botón para no enviar dos o más veces la misma instrucción mientras procesa
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=process_button
        ).then(
            fn=lambda x: main_apendice(files_paths=x,
                                       certificates_path=paths['certificates_folder'],  # Folder of saved certificates
                                       fonts_path=paths['fonts_folder'],                # Folder of saved fonts
                                       images_path=paths['images_folder'],              # Folder to save temporary images
                                       apendices_path=paths['apendices_folder'],        # Folder to save generated apendices
                                       progress=gr.Progress()
                                       ),
            inputs=file_input,                      # >List of certificate file names
            outputs=[output_text, zip_file_path]    # list of generated apendice names and zip path to download
        ).then(
            # Habilita el botón descarga cuando termina de procesar
            fn=lambda: gr.update(elem_id="download_button_apendices", interactive=True),
            inputs=None,
            outputs=zip_file_path
        )

        # Cleanning Function
        clean_button.click(
            fn=lambda: ("", None),  # Deja el Textbox vacío y resetea el FileInput
            inputs=None,
            outputs=[output_text, file_input]
        )
    # ---------------------------------------------------------------------------------------

    # # ---------------------------------------------------------------------------------------
    # TAB TESTIMONIOS
    # Generate Testimonio from Dittos and MS Excel file, append closing, pie y carátula
    # ---------------------------------------------------------------------------------------
    with gr.Tab("Testimonios") as testimonios_tab:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Modelo de Cierre configurado")
                closing_model_selected = gr.Text(
                    value=paths['closing_model_name'],      # Closing model name
                    lines=1,
                    container=False,
                    interactive=False,
                )

                gr.Markdown("Modelo de Pie configurado")
                pie_model_selected = gr.Text(
                    value=paths['pie_model_name'],          # Pie model name
                    lines=1,
                    container=False,
                    interactive=False,
                )

                gr.Markdown("Cargue los archivo .docx de las escrituras firmadas")
                file_input = gr.Files(label="",
                                       scale=1,
                                       file_count="multiple",
                                       type="filepath",
                                       height=270,
                                       )


                # input
                signature_date = gr.Textbox(label='Fecha de firma dd/mm/aaaa',
                                            elem_id='signature_date')

                # input
                testimonio_date = gr.Textbox(label='Fecha del Testimonio dd/mm/aaaa',
                                             elem_id='testimonio_date')

                # save date in config file
                signature_date.change(
                    fn=save_dates_config,
                    inputs=[signature_date, testimonio_date],
                    outputs=None,
                )

                testimonio_date.change(
                    fn=save_dates_config,
                    inputs=[signature_date, testimonio_date],
                    outputs=None,
                )

                # -------------------------------------------------------------------------------------
                process_button = gr.Button("Procesar", interactive=False)
                zip_file_path = gr.DownloadButton(elem_id="download_button_testimonios",
                                                  label="Descargar",
                                                  interactive=False)

                clean_button = gr.Button("Limpiar")

            with gr.Column(scale=8):
                gr.Markdown("Resultados")
                output_text = gr.Textbox(label="",
                                         lines=20,
                                         max_lines=20,
                                         autoscroll=False,
                                         )
        def change_process_button(file_input):
            return gr.update(interactive=file_input is not None)

        file_input.change(fn=change_process_button, inputs=file_input, outputs=process_button)

        # Process
        # Get testimonio margins configurations
        configurations = read_configuration_file('config')

        testimonio_margins = [
                                configurations['MARGEN_SUPERIOR_TESTIMONIO'],
                                configurations['MARGEN_INFERIOR_TESTIMONIO'],
                                configurations['MARGEN_IZQUIERDO_TESTIMONIO'],
                                configurations['MARGEN_DERECHO_TESTIMONIO'],
                                configurations['MARGEN_ENCUADERNACION_TESTIMONIO'],
                                configurations['TAB_POSICION_TESTIMONIO'],
                              ]
        dates = [
            configurations['FECHA_FIRMA'],
            configurations['FECHA_TESTIMONIO'],
        ]

        print(f'testimonio margins: {testimonio_margins}')
        print(f'testimonio dates: {dates}')

        # Call `main_ditto` utils function
        process_button.click(
            fn=lambda x: validate_date(x),
            inputs=signature_date,
            outputs=signature_date
        ).then(
            fn=lambda x: validate_date(x),
            inputs=testimonio_date,
            outputs=testimonio_date
        ).then(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=process_button
        ).then(
            # Call main_ditto utils function
            fn=lambda x: main_testimonio(instruments_paths=x,
                                         closing_model_path=paths['closing_model_path'],
                                         pie_model_path=paths['pie_model_path'],
                                         testimonio_margins=testimonio_margins,
                                         testimonios_path=paths['testimonios_folder'],
                                         signature_date=read_configuration_file('config')['FECHA_FIRMA'],
                                         testimonio_date=read_configuration_file('config')['FECHA_TESTIMONIO'],
                                         progress=gr.Progress(),
                                         ),
            inputs=file_input,                                              # input database file
            outputs=[output_text, zip_file_path]                            # list of generated dittos names, zip path to download
        ).then(
            # Enable Download button when process finish
            fn=lambda: gr.update(elem_id="download_button_testimonios", interactive=True),
            inputs=None,
            outputs=zip_file_path
        )

        # Clean textbox and file input components
        clean_button.click(
            fn=lambda: ("", None),
            inputs=None,
            outputs=[output_text, file_input]
        ).then(
            # Enable process button when cleaning finish
            fn=lambda: gr.update(elem_id="process_button", interactive=False),
            inputs=None,
            outputs=process_button
        ).then(
            # Unable download button when cleaning finish
            fn=lambda: gr.update(elem_id="download_button_testimonios", interactive=False),
            inputs=None,
            outputs=zip_file_path
        )
    # ---------------------------------------------------------------------------------------





    # ---------------------------------------------------------------------------------------
    # TAB CONFIGURATIONS
    # Configurations for Dittos and Testimonios
    # ---------------------------------------------------------------------------------------
    with gr.Tab("Configuraciones") as configs_tab:
        # -------------------------------------------------------------------
        # DITTO CONFIGURATIONS
        # -------------------------------------------------------------------
        with gr.Tab("Ditto") as ditto_config_tab:

            # Read current configuration
            configurations = read_configuration_file('config')

            with gr.Row():
                # COLUMN 1
                with gr.Column(scale=2):

                    # Get stored ditto models_dittos
                    ditto_models_list = get_dittos_models()

                    # Dropdown Ditto Model Selector
                    ditto_model_selector = gr.Dropdown(elem_id='dropdown_models_ditto',
                                                       label='Seleccionar modelo',
                                                       choices=ditto_models_list,
                                                       value=configurations['MODELO_ESCRITURA'],
                                                       interactive=True)

                    # Load Ditto Models Files
                    gr.Markdown("Cargue más modelos")
                    upload_ditto_models = gr.Files(label="Cargar modelo", type="filepath", height=1)

                    # Btn Eliminar Modelo
                    btn_remove_ditto_model = gr.Button("Eliminar modelo", scale=2)

                    # Btn Guardar Configuración
                    btn_save_config_ditto = gr.Button("Actualizar configuración")

                # COLUMN 2
                with gr.Column(scale=2):
                    pass

                # COLUMN 3
                with gr.Column(scale=2):
                    # Ditto Margins
                    # Top margin
                    top_margin_ditto = gr.Slider(elem_id='top_margin_ditto', label="Margen Superior",
                                                 value=configurations['MARGEN_SUPERIOR_DITTO'], minimum=1.0,
                                                 maximum=10.0, step=0.1, interactive=True)
                    # Bottom margin
                    bottom_margin_ditto = gr.Slider(elem_id='bottom_margin_ditto', label="Margen Inferior",
                                                    value=configurations['MARGEN_INFERIOR_DITTO'], minimum=1.0,
                                                    maximum=5.0, step=0.1, interactive=True)
                    # Left margin
                    left_margin_ditto = gr.Slider(elem_id='left_margin_ditto', label="Margen Izquierdo",
                                                  value=configurations['MARGEN_IZQUIERDO_DITTO'], minimum=1.0,
                                                  maximum=5.0, step=0.1, interactive=True)
                    # Right margin
                    right_margin_ditto = gr.Slider(elem_id='right_margin_ditto', label="Margen Derecho",
                                                   value=configurations['MARGEN_DERECHO_DITTO'], minimum=1.0,
                                                   maximum=5.0, step=0.1, interactive=True)
                    # Binding margin
                    binding_margin_ditto = gr.Slider(elem_id='binding_margin_ditto', label="Encuadernación",
                                                     value=configurations['MARGEN_ENCUADERNACION_DITTO'],
                                                     minimum=0.0, maximum=5.0, step=0.1, interactive=True)

                    # Tab Configurations
                    # Position
                    position_tab_ditto = gr.Slider(elem_id='tab_pos_ditto', label="Tabulación Posición",
                                                   value=configurations['TAB_POSICION_DITTO'], minimum=0.0,
                                                   maximum=20.0, step=0.25, interactive=True)

        # -------------------------------------------------------------------
        #  TESTIMONIOS CONFIGURATIONS
        # -------------------------------------------------------------------
        with gr.Tab("Testimonio") as testimonio_config_tab:
            # Get current configurations
            configurations = read_configuration_file('config')

            with gr.Row():
                with gr.Column(scale=2):
                    # Get stored closing models_closing
                    closing_models_list = get_closing_models()
                    # Closing Models Selector
                    closing_model_selector = gr.Dropdown(elem_id='dropdown_models_closing',
                                                         label='Seleccionar modelo de cierre',
                                                         choices=closing_models_list,
                                                         value=configurations['MODELO_CIERRE'],
                                                         interactive=True)

                    # Load Closing Models
                    gr.Markdown("Cargue más plantillas de cierres")
                    upload_closing_models = gr.Files(label="Cargar modelos de cierre", type="filepath", height=1)

                    # Btn Remove Closing Model
                    btn_remove_closing_model = gr.Button('Eliminar cierre')

                    # Btn Save Testimonio Configurations
                    btn_save_config_testimonio = gr.Button("Actualizar configuración")

                with gr.Column(scale=2):
                    # Get stored pie in models_pie
                    pie_models_list = get_pies_models()

                    # Pie Model Selector
                    pie_model_selector = gr.Dropdown(elem_id='dropdown_models_pie',
                                                         label='Seleccionar modelo de pie',
                                                         choices=pie_models_list,
                                                         value=configurations['MODELO_PIE'],
                                                         interactive=True)

                    # Load Pies Models
                    gr.Markdown("Cargue más plantillas de pies")
                    upload_pie_models = gr.Files(label="Cargar modelos de pie", type="filepath", height=1)

                    # Btn Remove Pie
                    btn_remove_pie_model = gr.Button('Eliminar pie')

                with gr.Column(scale=2):
                    # Margins configurations
                    # Top margin
                    top_margin_testimonio = gr.Slider(label="Margen Superior",
                                                      value=configurations['MARGEN_SUPERIOR_TESTIMONIO'], minimum=1.0,
                                                      maximum=10.0, step=0.1, interactive=True)
                    # Bottom margin
                    bottom_margin_testimonio = gr.Slider(label="Margen Inferior",
                                                         value=configurations['MARGEN_INFERIOR_TESTIMONIO'], minimum=1.0,
                                                         maximum=5.0, step=0.1, interactive=True)
                    # Left margin
                    left_margin_testimonio = gr.Slider(label="Margen Izquierdo",
                                                       value=configurations['MARGEN_IZQUIERDO_TESTIMONIO'], minimum=1.0,
                                                       maximum=5.0, step=0.1, interactive=True)
                    # Right margin
                    right_margin_testimonio = gr.Slider(label="Margen Derecho",
                                                        value=configurations['MARGEN_DERECHO_TESTIMONIO'], minimum=1.0,
                                                        maximum=5.0, step=0.1, interactive=True)
                    # Binding margin
                    binding_margin_testimonio = gr.Slider(label="Encuadernación",
                                                          value=configurations['MARGEN_ENCUADERNACION_TESTIMONIO'],
                                                          minimum=0.0, maximum=5.0, step=0.1, interactive=True)

                    # Tab configurations
                    # Position
                    position_tab_testimonio = gr.Slider(label="Tab Posición",
                                                        value=configurations['TAB_POSICION_TESTIMONIO'], minimum=0.0,
                                                        maximum=20.0, step=0.25, interactive=True)


        # Update Ditto and Testimonio Tabs with selected models
        dittos_tab.select(lambda: os.path.basename(get_paths()['ditto_model_name']), None, ditto_model_selected)
        testimonios_tab.select(lambda: os.path.basename(get_paths()['closing_model_name']), None, closing_model_selected)
        testimonios_tab.select(lambda: os.path.basename(get_paths()['pie_model_name']), None, pie_model_selected)

    # --------------------------------------
    # EVENTS TAB CONFIG DITTO
    # --------------------------------------
    def clean_files_upload(_):
        # Return None to clean gr.File
        return
    # Update Ditto Model Selector
    # --------------------------------------
    upload_ditto_models.change(
        fn=load_ditto_models,               # Call function to load dittos
        inputs=upload_ditto_models,         # input: file list of selected models by the user
        outputs=None,
        postprocess=True,
    ).then(
        fn=update_ditto_model_selector,   # Call function to update the files list
        inputs=None,
        outputs=ditto_model_selector,       # Return the selector control
        postprocess=True,
    ).then(
        fn=clean_files_upload,              # Call Clean function
        inputs=upload_ditto_models,         # Input the upload control
        outputs=upload_ditto_models,        # Return the upload control
        postprocess=True,
    )


    # Remove Selected Ditto Model
    # --------------------------------------
    btn_remove_ditto_model.click(
        fn=remove_selected_ditto,
        inputs=ditto_model_selector,
        outputs=ditto_model_selector
    ).then(
        fn=update_ditto_model_selector,
        inputs=None,
        outputs=ditto_model_selector
    )


    # Save config Ditto
    # --------------------------------------
    def update_selected_model(tbx_model_selected):
        return

    btn_save_config_ditto.click(
        fn=save_ditto_config,
        inputs=[
                ditto_model_selector,
                top_margin_ditto,
                bottom_margin_ditto,
                left_margin_ditto,
                right_margin_ditto,
                binding_margin_ditto,
                position_tab_ditto,
                ],
        outputs=None,
    ).then(
        fn=None,
        inputs=None,
        outputs=ditto_model_selector,
    ).then(
        fn=update_selected_model,
        inputs=ditto_model_selected,
        outputs=ditto_model_selected,
    )

    # Update textbox ditto selected model
    gr.update(value=paths['ditto_model_name'])

    # --------------------------------------
    # EVENTS TAB CONFIG TESTIMONIO
    # --------------------------------------
    # CLOSING & PIE MODELS EVENTS
    # Update Dropdown Closing Models Selector
    # --------------------------------------
    upload_closing_models.change(
        fn=load_closing_models,
        inputs=upload_closing_models,
        outputs=None,
        postprocess=True
    ).then(
        fn=update_closing_model_selector,
        inputs=None,
        outputs=closing_model_selector,
        postprocess=True,
    ).then(
        fn=clean_files_upload,
        inputs=upload_closing_models,
        outputs=upload_closing_models,
        postprocess=True,
    )

    # Update Pies Models Selector
    # --------------------------------------
    upload_pie_models.change(
        fn=load_pie_models,
        inputs=upload_pie_models,
        outputs=None,
        postprocess=True
    ).then(
        fn=update_pie_model_selector,
        inputs=None,
        outputs=pie_model_selector,
        postprocess=True,
    ).then(
        fn=clean_files_upload,
        inputs=upload_pie_models,
        outputs=upload_pie_models,
        postprocess=True,
    )

    # Remove Selected Closing Model
    # --------------------------------------
    btn_remove_closing_model.click(
        fn=remove_selected_closing,
        inputs=closing_model_selector,
        outputs=closing_model_selector
    ).then(
        fn=update_closing_model_selector,
        inputs=None,
        outputs=closing_model_selector
    )

    # Remove Selected Pie Model
    # --------------------------------------
    btn_remove_pie_model.click(
        fn=remove_selected_pie,
        inputs=pie_model_selector,
        outputs=pie_model_selector
    ).then(
        fn=update_pie_model_selector,
        inputs=None,
        outputs=pie_model_selector
    )



    # Save config Testimonio
    # --------------------------------------
    def update_selected_model(tbx_model_selected):
        return

    btn_save_config_testimonio.click(
        fn=save_testimonio_config,
        inputs=[
                closing_model_selector,
                pie_model_selector,
                top_margin_testimonio,
                bottom_margin_testimonio,
                left_margin_testimonio,
                right_margin_testimonio,
                binding_margin_testimonio,
                position_tab_testimonio,
                ],
        outputs=None,
    ).then(
        fn=None,
        inputs=None,
        outputs=ditto_model_selector,
    ).then(
        fn=update_selected_model,
        inputs=ditto_model_selected,
        outputs=ditto_model_selected,
    )

    # Update textbox ditto selected model
    gr.update(value=paths['ditto_model_name'])


    # ---------------------------------------------------------------------------------------
    # server_name='192.168.1.59',
# Lanzar la interfaz
if __name__ == '__main__':
    app.launch(
        share=False,
        debug=False,
        server_name="192.168.1.159",
        server_port=8080,
        # ssl_keyfile="server.key",
        # ssl_certfile="server.crt",
        # ssl_verify=False,
    )
