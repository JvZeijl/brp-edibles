import os, os.path as path

DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'out'

if not path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)