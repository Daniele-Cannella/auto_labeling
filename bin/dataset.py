import os
import json
from image import Image
from alias import Alias


class Dataset():
    def __init__(self):
        self.images = []
        self.aliases = []

    def _add_image(self, image:object):
        if isinstance(image, Image):
            self.images.append(image)
        else:
            raise ValueError("Errore: l'oggetto non è di tipo Image")

    def _add_alias(self, alias:object):
        self.aliases.append(alias)

    def save_data(self):
        json_file = '../data/dataset.json'

        try:
            with (open(json_file, 'a') as f):
                self.images = [str(image) for image in self.images]
                for image in self.images:
                    data = {
                        'images_saves' : image,
                    }
                json.dump(data, f, indent=4)

                for alias in self.aliases:
                    data = {
                        'alias' : alias.alias,
                        'class_id' : alias.class_id,
                        'metrics' : alias.metrics
                    }
                json.dump(data, f, indent=4)
                f.write("\n")

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
