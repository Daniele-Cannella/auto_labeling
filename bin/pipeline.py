'''
Flow of the pipeline:
1. Find path of the images and create a list of Image objects.
2. Load the images using the specified library.
3. Ask the LLm for alias of the classes.
4. use the alias generated to predict the bbox of the images.

'''

import os
import concurrent.futures

from image import Image
from metrics import Metrics
from model_llm import generate_text
from alias import Alias
from dataset import Dataset
from model_vis import ModelVis

from log import Log
from icecream import ic
import argparse


classes = {
    "bitter pack" : 0,
    "bottle pack" : 1,
    "box" : 2,
    "can pack" : 3,
    "crate" : 4,
    "keg" : 5,
}


def process_class(list_of_images: list[Image], class_name: str, num_alias: int):
    for num in range(num_alias):
        try:
            alias = generate_text(class_name)
            ic(f"Generated alias for class {class_name}: {alias}")
        except Exception as e:
            ic(f"Error generating text: {e}")
            logger.write_error(e)
            continue

        list_of_gt = [image.get_ground_truth() for image in list_of_images]

        model_vis = ModelVis(list_of_images, list_of_gt)
        confidence_scores = model_vis.predict(alias)

        metrics = Metrics(confidence_scores)
        mean_auc_score, results = metrics.get_precision_recall()

        alias = Alias(alias, classes[class_name], results)

    pass


def args_parsing():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--indir', type=str, help='Input image path')
    parser.add_argument('-a', '--alias', type=int, help='Num of alias for each class')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    return args


def batching_images(image_path_list: list[str], batch_size: int):
    for i in range(0, len(image_path_list), batch_size):
        yield image_path_list[i:i + batch_size]


def main(logger: object):
    args = args_parsing()

    if args.verbose:
        ic.enable()
    else:
        ic.disable()

    dataset = Dataset()
    images_path_list = dataset.images_dir(args.indir)   # '../data/images'

    # c = batching_images(images_path_list, 10)
    # print(next(c))
    # print(next(c))

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


if __name__ == "__main__":
    nome_f = os.path.basename(__file__)
    logger = Log(nome_f)
    logger.log(False)
    main(logger)
    logger.log(True)

