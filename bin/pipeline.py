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
from icecream import ic

ic.enable()

'''
class Pipeline:
    def __init__(self, logger: object):
        self.logger = logger
'''


def main(logger: object):
   dataset = Dataset()
   images_path_list = dataset.images_dir('../data/images')
   image_list = [Image(image_path) for image_path in images_path_list]  # Create a list of Image objects
   
   for image in image_list:
       image.load('PIL')
       # ic(image.image)

    
    
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

def process_chunk(chunk):
    
    Processes a chunk of images and returns a list of results.
    :param chunk: List of image paths to process.
    :return: List of tuples containing testo, confusion_matrix, and percorso for each image.
    
    results = []
    for percorso in chunk:
        img = Image(percorso)
        gt = img.gt
        testo = LLM.get_testo
        results_vis = modello_vis(img, testo)

        confusion_matrix = Processing(results_vis, gt).confusion_matrix
        results.append((testo, confusion_matrix, percorso))
    return results

def chunk_list(lst, n):
    
    Splits a list into n nearly equal chunks.
    :param lst: List to split.
    :param n: Number of chunks.
    :return: List of chunks.
    
    chunk_size = ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main(lista_immagini):
    
    Main function to process a list of images using multiprocessing with even split among cores.
    :param lista_immagini: List of image paths to process.
    
    dataset = Dataset()
    num_cores = os.cpu_count() or 1  # Ensure at least 1 core

    # Split lista_immagini into chunks for each core
    chunks = chunk_list(lista_immagini, num_cores)

    all_results = []

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks and collect results
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
            except Exception as e:
                print(f"Error processing chunk: {e}")

    # Iterate through the combined results
    for testo, confusion_matrix, percorso in all_results:
        if testo and confusion_matrix:  # Ensure valid results
            metrics = Metrics(confusion_matrix)
            average_precision = metrics.av_p
            alias = Alias(testo, average_precision)
            dictionary = alias.dictionary
            dataset.save_data(dictionary)

    # Load the combined data if needed
    dict = dataset.load('path_to_json')
    print(dict)


if __name__ == "__main__":
    lista_immagini = ['/path/to/image1.jpg', '/path/to/image2.jpg']  # Replace with actual image paths
    main(lista_immagini)

"""
