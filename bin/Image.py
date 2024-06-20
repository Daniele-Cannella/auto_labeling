import cv2
from PIL import Image as Pil_Image
import pathlib


class Image:
    def __init__(self, path):
        self.image = None
        self.path = pathlib.Path(path)
        self.gt = self.load_gt()

    def load(self, load_type):
        str_path = str(self.path)
        if load_type.lower() == "opencv":
            self.image = cv2.imread(str_path)
        elif load_type.lower() == "pil" or load_type.lower() == "pillow":
            self.image = Pil_Image.open(str_path)
        else:
            raise ValueError("Unsupported load type. Use 'opencv' or 'pil'.")

    def load_gt(self):
        filename = self.path.stem

        path = self.path.parent / (filename + ".txt")

        result = []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.split(" ")
                class_id = line[0]
                bbox = line[1:]
                result.append((int(class_id), [float(coord) for coord in bbox]))
        return result

    def __str__(self):
        image_info = f"Image path: {self.path}\n"
        image_info += f"Image size: {self.image.size}\n" if self.image else "Image size: Not loaded\n"
        gt_info = f"Ground truth data:\n"
        for class_id, bbox in self.gt:
            gt_info += f"Class ID: {class_id}, Bbox: {bbox}\n"
        return image_info + gt_info

