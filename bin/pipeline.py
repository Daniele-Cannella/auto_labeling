'''
Flow of the pipeline:
1. Find path of the images and create a list of Image objects.
2. Load the images using the specified library.
3. Ask the LLm for alias of the classes.
4. use the alias generated to predict the bbox of the images.

'''

from image import Image
from metrics import Metrics
from model_llm import generate_text, LLM
from alias import Alias
from dataset import Dataset
from model_vis import ModelVis

from log import Log
import os
import concurrent.futures
from icecream import ic
import argparse


'''
class Pipeline:
    def __init__(self, logger: object):
        self.logger = logger
'''

classes = ['class1', 'class2', 'class3', 'class4', 'class5']


def process_class(list_of_images: list[Image], class_name: str, num_alias: int):
    for num in range(num_alias):
        try:
            text = generate_text(class_name)
            ic(f"Generated text for class {class_name}: {text}")
        except Exception as e:
            ic(f"Error generating text: {e}")
            logger.write_error(e)
            continue

        confidence_scores = ModelVis(list_of_images)
        metrics = Metrics(confidence_scores)
        mean_auc_score, results = metrics.get_precision_recall()

        alias = Alias()
        alias.add_alias(class_name, text, mean_auc_score, results)
        

    pass


def args_parsing():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--indir', type=str, help='Input image path')
    parser.add_argument('-a', '--alias', type=int, help='Num of alias for each class')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    return args


def main(logger: object):
    args = args_parsing()

    if args.verbose:
        ic.enable()
    else:
        ic.disable()

    dataset = Dataset()
    images_path_list = dataset.images_dir(args.indir)   # '../data/images'
    image_list = [Image(image_path) for image_path in images_path_list]  # Create a list of Image objects
   
    for image in image_list:
        try:
            image.load('jpeg4py')
            ic(image.image)
        except AttributeError as e:
            print(f"Error loading image: {e}")
            logger.write_error(e)
            break
        except Exception as e:
            print(f"Error loading image: {e}")
            logger.write_error(e)
            continue

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_class, image_list, class_name, args.alias) for class_name in classes]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.write_error(f"Exception during class processing: {e}")


    pass


if __name__ == "__main__":
    nome_f = os.path.basename(__file__)
    logger = Log(nome_f)
    logger.log(False)
    main(logger)
    logger.log(True)

