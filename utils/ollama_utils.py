import json
import subprocess
import requests
import time
import socket

def kill_ollama_server(wait_time: int = 3, verbose=False):
    """
    Identifica el PID del servidor de Ollama y, si está activo, lo detiene usando `kill -9`.
    Espera un tiempo configurable para que la memoria de la GPU se libere.
    Parámetros:
        wait_time (int): Tiempo de espera en segundos antes de salir de la función (por defecto es 2).
    """
    try:
        # Buscar el PID del proceso de Ollama
        result = subprocess.run(
            ["pgrep", "-f", "ollama"],  # Busca solo el proceso "ollama serve"
            capture_output=True,
            text=True
        )
        # Verificar si se encontró un PID
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")  # Obtener todos los PIDs
            # Construir el comando con los PIDs como argumentos separados
            command = ["kill", "-9"] + pids
            try:
                if verbose:
                    print(f"Deteniendo procesos: {', '.join(pids)}.")
                subprocess.run(command, check=True)  # Enviar señal kill -9
                if verbose:
                    print("Procesos detenidos correctamente.")
            except subprocess.CalledProcessError as e:
                print(f"Error al detener los procesos con PIDs {pids}: {e}")

            # Esperar el tiempo configurado para que la GPU libere la memoria
            if verbose:
                print(f"Liberando la memoria de la GPU, esperar {wait_time} segundos.")
            time.sleep(wait_time)
        else:
            if verbose:
                print("El servidor de Ollama no está en ejecución.")
    except Exception as e:
        print(f"Error inesperado: {e}")

def _is_port_open(port=11434):
    """
    Verifica si un puerto específico está abierto en localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        result = s.connect_ex(("localhost", port))
        return result == 0  # Retorna True si el puerto está abierto

def _wait_for_ollama_server(timeout=10):
    """
    Espera hasta que el servidor de Ollama esté listo para aceptar conexiones.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _is_port_open():
            # print("Servidor de Ollama listo.")
            return True
        time.sleep(1)  # Espera un segundo antes de intentar nuevamente
    print("Tiempo de espera excedido. El servidor de Ollama no está listo.")
    return False

def start_ollama_server():
    """
    Inicia el servidor de Ollama en segundo plano si no está en ejecución.
    """
    if not _is_port_open():
        # print("Iniciando el servidor de Ollama en segundo plano.")
        try:
            # Inicia el servidor en segundo plano
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,  # No captura la salida estándar
                stderr=subprocess.DEVNULL,  # No captura la salida de error
                text=True
            )
            # Espera hasta que el servidor esté listo
            if _wait_for_ollama_server(timeout=10):  # Tiempo de espera configurable
                # print("Servidor de Ollama iniciado correctamente.")
                return
            else:
                raise ValueError("Error: Tiempo de espera excedido. El servidor de Ollama no está listo.")
        except FileNotFoundError:
            raise ValueError("Error: El comando 'ollama' no se encontró. Asegúrate de que Ollama esté instalado y en el PATH.")
        except PermissionError:
            raise ValueError("Error: No tienes permisos para ejecutar el comando 'ollama'.")
        except Exception as e:
            raise ValueError(f"Error inesperado al iniciar el servidor de Ollama: {e}.")
    else:
        # print("Servidor de Ollama ya está corriendo.")
        return


def _test_model_response(model_name):
    try:
        response = requests.post('http://localhost:11434/api/generate',
                                 json={
                                     "model": model_name,
                                     "prompt": "Hola",
                                     "stream": True
                                 },
                                 stream=True)

        if response.status_code == 200:
            # print(f"Respuesta del modelo {model_name}:")
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    chunk_data = json.loads(chunk)
                    if 'response' in chunk_data:
                        # print(chunk_data['response'], end='', flush=True)
                        full_response += chunk_data['response']

            if len(full_response.strip()) > 0:
                return True
        return False
    except Exception as e:
        print(f"Error al probar el modelo: {e}")
        return False

def run_ollama_model(model_name: str = None):
    """
    Inicia el modelo de Ollama, verifica que esté listo y prueba una respuesta simple.
    """
    if not model_name:
        # print("Error: No se proporcionó un nombre de modelo.")
        return False

    # print(f"Cargando el modelo '{model_name}'...")
    try:
        # Inicia el modelo en un proceso separado
        process = subprocess.Popen(
            ["ollama", "run", f"{model_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )

        # Verifica si el servidor está listo
        if _wait_for_ollama_server(timeout=10):
            # print(f"El modelo '{model_name}' se inició correctamente.")

            # Prueba una respuesta simple
            if _test_model_response(model_name):
                # print(f"El modelo '{model_name}' respondió correctamente a una prueba simple.")
                return True
            else:
                # print(f"Error: El modelo '{model_name}' no respondió correctamente a una prueba simple.")
                process.terminate()
                return False
        else:
            # print(f"Error: Tiempo de espera excedido. El modelo '{model_name}' no está listo.")
            process.terminate()
            return False

    except FileNotFoundError:
        # print("Error: El comando 'ollama' no se encontró. Asegúrate de que Ollama esté instalado y en el PATH.")
        pass
    except PermissionError:
        # print("Error: No tienes permisos para ejecutar el comando 'ollama'.")
        pass
    except Exception as e:
        # print(f"Error inesperado al iniciar el modelo '{model_name}': {e}.")
        pass
    return False



# Test the functions start and kill.
if __name__ == '__main__':

    for i in range(5):
        start_ollama_server()
        time.sleep(2)
        run_ollama_model(model_name='qwen2.5-max:14b')
        time.sleep(10)
        kill_ollama_server(wait_time=2)
        print('-'*30)
