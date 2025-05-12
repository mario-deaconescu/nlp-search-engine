from concurrent.futures import ProcessPoolExecutor

executor: ProcessPoolExecutor

def __init__worker():
    """
    Initialize the worker process.
    This function is called in the worker process to set up any necessary resources.
    """
    # This is where you can initialize any resources needed for the worker process
    return 0
