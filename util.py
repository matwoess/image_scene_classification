# -*- coding: utf-8 -*-
import os
import zipfile
from pathlib import Path

from PIL import Image


def zip_folder(folder_to_zip: Path, zipfile_name: str) -> None:
    with zipfile.ZipFile(f'{zipfile_name}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(folder_to_zip):
            for file in files:
                archive.write(os.path.join(root, file))


def get_image_shape(image_path: Path):
    return Image.open(image_path).size


def write_dict_to_file(dictionary: dict, file_path: Path):
    with open(file_path, 'w') as f:
        for key, value in dictionary.items():
            print(f'{key}: {value}', file=f)
