'''
Flow of the pipeline:
1. Find path of the images and create a list of Image objects.
'''

from image import Image
from metrics import Metrics
from model_llm import ModelLM
from alias import Alias
from dataset import Dataset
from log import Log
import os
import concurrent.futures
from math import ceil

'''
class Pipeline:
    def __init__(self, logger: object):
        self.logger = logger
'''


def main(logger: object):
   dataset = Dataset()
   images_path_list = dataset.images_dir('../data/images')
   image_list = [Image(image_path) for image_path in images_path_list]  # Create a list of Image objects
   pass


if __name__ == "__main__":
    nome_f = os.path.basename(__file__)
    logger = Log(nome_f)
    logger.log(False)
    main(logger)
    logger.log(True)
"""
import concurrent.futures
import os
from math import ceil
from python_classes import Image, LLM, Processing, Metrics, Alias, Dataset



if __name__ == "__main__":
    lista_immagini = ['/path/to/image1.jpg', '/path/to/image2.jpg']  # Replace with actual image paths
    main(lista_immagini)

"""
