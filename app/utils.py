import os

def ensure_model_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
