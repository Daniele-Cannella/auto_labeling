import sys
import math
import cv2 as cv
import numpy as np
import os

import concurrent.futures


def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi


def get_lines(image_path, output_path):
    # default_file = '../data/images/Horizontal/0ece813e-4200080352_1108_156.jpg'   #   0b7d98ea-4200080346_5543_156.jpg
    filename = image_path
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + image_path + '] \n')
        return -1

    '''
    height, width = src.shape[:2]
    new_dimensions = (width // 4, height // 4)  # New dimensions are half the original dimensions
    resized_src = cv.resize(src, new_dimensions, interpolation=cv.INTER_AREA)
    '''
    dst = cv.Canny(src, 150, 250, None, 3)  # threshold?? 50, 200 

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 300, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 500, None, 100, 50)

    desired_angle = 9   # 10, 12, 15
    angle_tolerance = 10   # 10

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1, y1, x2, y2 = l
            angle = calculate_angle(x1, y1, x2, y2)
            if abs(angle - desired_angle) < angle_tolerance or abs(angle + 180 - desired_angle) < angle_tolerance:
                cv.line(cdstP, (x1, y1), (x2, y2), (0,0,255), 2, cv.LINE_AA)

    # cv.imshow("Source", src)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    # 
    # cv.waitKey()

    cv.imwrite(output_path, cdstP)  #   0b7d98ea-4200080346_5543_156_houghlines.jpg

def process_image(image_path, output_path):
    get_lines(image_path, output_path)
    print("Processed image: ", image_path)

max_workers = 8

def process_images(image_dir, output_dir):
    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        for image in os.listdir(image_dir):
            if not image.endswith('.jpg'):
                continue
            image_path = os.path.join(image_dir, image)
            output_path = os.path.join(output_dir, image)
            executor.submit(process_image, image_path, output_path)


def main():
    image_dir = "../data/images/Horizontal/"
    output_dir = '../data/images/Horizontal_houghlines_images/'
    process_images(image_dir, output_dir)  


if __name__ == "__main__":
    main()