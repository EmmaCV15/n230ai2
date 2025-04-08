import GPUtil

def gpu_checkup():

    """Verificar si una GPU está disponible y es utilizable."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus
    else:
        return False

import GPUtil
import time

def wait_for_gpu(required_memory=7300, gpu_id=0, sleep_duration=1):
    """Waits until the specified GPU has enough free memory to start processing.

    Args:
        required_memory (int, optional): The amount of free memory required in MB. Defaults to 4000.
        gpu_id (int, optional): The ID of the GPU to monitor. Defaults to 0.
        sleep_duration (int, optional): How long to sleep (in seconds) between checks. Defaults to 60.

    Returns:
        None
    """
    while True:
        try:
            GPUs = GPUtil.getGPUs()
            if not GPUs:
                continue

            gpu = GPUs[gpu_id]
            available_memory = gpu.memoryFree

            if available_memory >= required_memory:
                break
            else:
                time.sleep(sleep_duration)
        except Exception as e:
            print(f"Error checking GPU memory: {e}.")


# Test functions
if __name__ == '__main__':
    gpus = gpu_checkup()
    for gpu in gpus:
        print(f'MemFree: {gpu.memoryFree}')
        print(f"Device: {gpu.name} driver: {gpu.driver} total memory: {gpu.memoryTotal} MB")
        print(f'Memory used: {gpu.memoryUsed} MB {(gpu.memoryUsed/gpu.memoryTotal)*100:.1f}%')

    wait_for_gpu(required_memory=7300, gpu_id=0, sleep_duration=1)
    print("Starting GPU processing...")
    # Your GPU processing code here
