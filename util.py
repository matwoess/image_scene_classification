import os
import zipfile
from pathlib import Path

from PIL import Image


def zip_folder(folder_to_zip: Path, zipfile_name: str) -> None:
    archive = zipfile.ZipFile(f'{zipfile_name}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder_to_zip):
        for file in files:
            archive.write(os.path.join(root, file))
    archive.close()


def get_image_shape(image_path: Path):
    image = Image.open(image_path)
    return image.size


def write_dict_to_file(dictionary: dict, file_path: Path):
    with open(file_path, 'w') as f:
        for key, value in dictionary.items():
            print(f'{key}: {dictionary[key]}', file=f)
