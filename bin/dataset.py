import os
import json
from image import Image


class Dataset():
    images = []
    aliases = []

    def __init__(self, main_dir: str):
        self.main_dir = main_dir

    def __dict__(self):
        Dataset.images = [str(image) for image in Dataset.images]
        return {
            'images': self.images
        }

    def _add_image(self, image):
        if isinstance(image, Image):
            self.images.append(image)
            print(f"Image {image} added to dataset")
        else:
            raise ValueError("Errore: l'oggetto non è di tipo Image")

    def _add_alias(self, alias):
        if alias is type(Alias):
            self.list_aliases.append(alias)
        else:
            raise ValueError("Errore: l'oggetto non è di tipo Alias")

    @classmethod
    def save_data(cls):
        json_file = './dataset/dataset.json'

        with open(json_file, 'a') as f:
            Dataset.images = [str(image) for image in Dataset.images]

            for image in Dataset.images:
                data = {
                    f'images_saves' : Dataset.images,
                    f'aliases_saves' : Dataset.aliases
                }
            json.dump(data, f, indent=4)

    def load_data(self):
        json_file = './dataset/dataset.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def images_dir(self, dir:str):
        list_images_path = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    list_images_path.append(os.path.join(root, file))

        return list_images_path
