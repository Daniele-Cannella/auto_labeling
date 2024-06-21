import cv2
from PIL import Image as Pil_Image
import pathlib
import numpy as np


class Image:
    def __init__(self, path: str, logger: object) -> None:
        self.image = None
        self.path = pathlib.Path(path)
        self.gt = self.load_gt()
        self.logger = logger

    def load(self, load_type: str) -> None:
        """
        Load the image using the specified library.

        :param load_type: The type of library to use for loading the image. Use 'opencv' or 'pil'.
        :return: None
        """
        str_path = str(self.path)
        if load_type.lower() == "opencv":
            self.image = cv2.imread(str_path)
        elif load_type.lower() == "pil" or load_type.lower() == "pillow":
            self.image = Pil_Image.open(str_path)
        else:
            raise ValueError("Unsupported load type. Use 'opencv' or 'pil'.")

    def load_gt(self) -> list[tuple[int, list[float]]]:
        """
        Load ground truth data from a text file associated with the image.

        :return: A list of tuples where each tuple contains a class ID and a list of bounding box coordinates.
        """
        filename = self.path.stem
        path = self.path.parent / (filename + ".txt")

        result = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.split()
                    class_id = int(line[0])
                    bbox = [float(coord) for coord in line[1:]]
                    result.append((class_id, bbox))
        except FileNotFoundError:
            print(f"Ground truth file not found: {path}")
        except Exception as e:
            print(f"Error loading ground truth data: {e}")

        return result

    def __str__(self) -> str:
        """
        Nicely format the image object information.

        :return: A string representation of the image object.
        """
        image_info = f"Image path: {self.path}\n"
        if isinstance(self.image, Pil_Image.Image):
            image_info += f"Image size (PIL): {self.image.size}\n"
        elif isinstance(self.image, np.ndarray):
            height, width = self.image.shape[:2]
            image_info += f"Image size (OpenCV): {width}x{height}\n"
        else:
            image_info += "Image size: Not loaded\n"

        gt_info = "Ground truth data:\n"
        for class_id, bbox in self.gt:
            gt_info += f"Class ID: {class_id}, Bbox: {bbox}\n"
        return image_info + gt_info
