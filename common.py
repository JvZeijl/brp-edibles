import os, os.path as path
from typing import Iterable, TypeVar

# Folder structure
DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'out'

if not path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

# Progressbar with fallback if tqdm is not installed
_T = TypeVar("T")
def tqdm(x: Iterable[_T], desc: str = ""):
    try:
        from tqdm import tqdm

        return tqdm(x, desc)
    except ImportError:
        return x

# File utilities
def list_files(dir_path: str) -> list[str]:
    return [
        path.join(dir_path, entry) for entry in os.listdir(dir_path)
            # Ignore directories and hidden files
            if path.isfile(path.join(dir_path, entry)) and not entry.startswith('.')
    ]

def list_directories(dir_path: str) -> list[str]:
    return [
        path.join(dir_path, entry) for entry in os.listdir(dir_path)
            # Ignore files and hidden directories
            if path.isdir(path.join(dir_path, entry)) and not entry.startswith('.')
    ]
