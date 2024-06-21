import os
import json
from image import Image
from alias import Alias
from log import Log


class Dataset():
    images = []
    aliases = []

    def __init__(self, main_dir: str = None):
        self.main_dir = main_dir


    def __dict__(self):
        Dataset.images = [str(image) for image in Dataset.images]
        return {
            'images': self.images
        }

    def _add_image(self, image: object):
        if isinstance(image, Image):
            self.images.append(image)
            print(f"Image {image} added to dataset")
        else:
            raise ValueError("Errore: l'oggetto non è di tipo Image")

    def _add_alias(self, alias: object):
        if alias is type(Alias):
            self.aliases.append(alias)
        else:
            raise ValueError("Errore: l'oggetto non è di tipo Alias")

    @classmethod
    def save_data(cls):
        json_file = '../data/dataset.json'

        try:
            with open(json_file, 'a') as f:
                Dataset.images = [str(image) for image in Dataset.images]

                for image in Dataset.images:
                    data = {
                        'images_saves' : Dataset.images,
                        'aliases_saves' : Dataset.aliases
                    }
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_data(self) -> list[dict]:
        try:
            json_file = '../data/dataset.json'
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def images_dir(self, dir:str) -> list[str]:
        list_images_path = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    list_images_path.append(os.path.join(root, file))

        return list_images_path
