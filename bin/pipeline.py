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
from model_vis import request_vis

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


def process_class(dataset: Dataset, list_of_images: list[Image], class_name: str, num_alias: int, groq: bool, apikey: str):
    # global alias, results
    c = 0
    for num in range(num_alias):
        try:
            alias = generate_text(class_name, groq, apikey)
            print(f"Generated alias for class {class_name}: {alias}")
        except Exception as e:
            ic(f"Error generating text: {e}")
            logger.write_error(e)
            continue

        # list_of_gt = [image.get_ground_truth() for image in list_of_images]

        result = request_vis(alias)
        print(result[1])

        metrics = Metrics(result[1]['data'])
        mean_auc_score, results = metrics.get_precision_recall()

        dataset._add_alias(Alias(alias, classes[class_name], results))
        if c == num_alias:
            print("Alias generated for class: ", class_name)
        
        c += 1
    return


def args_parsing():
    parser = argparse.ArgumentParser(
        description='Process images to generate class aliases and predict bounding boxes using a visual model.'
    )
    parser.add_argument('-i', '--indir', type=str, help='Input image path')
    parser.add_argument('-a', '--alias', type=int, help='Num of alias for each class')
    parser.add_argument('-q', '--groq', action="store_true", default=False, help='Use groq for prediction')
    parser.add_argument('-k', '--apikey', type=str, help='API key for the LLM Groq')
    parser.add_argument('--read-type', type=str, help='Type of read for images (jpeg4py, opencv, pil)')
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
            image.load(args.read_type)
            dataset._add_image(image)
            ic(image.image)
        except AttributeError as e:
            print(f"Error loading image: {e}")
            logger.write_error(e)
            break
        except Exception as e:
            print(f"Error loading image: {e}")
            logger.write_error(e)
            continue
    
    '''
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_class, dataset, image_list, class_name, args.alias) for class_name in classes]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.write_error(f"Exception during class processing: {e}")

    '''
    for class_name in classes:
        print(f"Processing class: {class_name}")
        process_class(dataset, image_list, class_name, int(args.alias), args.groq, args.apikey)

    dataset.save_data()


if __name__ == "__main__":
    nome_f = os.path.basename(__file__)
    logger = Log(nome_f)
    logger.log(False)
    main(logger)
    logger.log(True)

