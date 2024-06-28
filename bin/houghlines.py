import cv2 as cv
import numpy as np
import os
import concurrent.futures

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def get_lines(image_path, output_path):
    src = cv.imread(cv.samples.findFile(image_path), cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f'Error opening image: {image_path}')
        return None
    
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)
    contrasted = cv.convertScaleAbs(src, alpha=alpha, beta=beta)

    dst = cv.Canny(contrasted, 150, 250, None, 3)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 500, None, 100, 50)

    good_lines = []
    desired_angle = 10
    angle_tolerance = 10

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1, y1, x2, y2 = l
            angle = calculate_angle(x1, y1, x2, y2)
            if abs(angle - desired_angle) < angle_tolerance or abs(angle + 180 - desired_angle) < angle_tolerance:
                if image_path == "../data/images/Horizontal/0b7d98ea-4200080346_5543_156.jpg":
                    good_lines.append(l)
                cv.line(cdstP, (x1, y1), (x2, y2), (0,0,255), 2, cv.LINE_AA)

    cv.imwrite(output_path, cdstP)
    return good_lines

def process_image(image_path, output_path):
    good_lines = get_lines(image_path, output_path)
    print(f"Processed image: {image_path}")
    return good_lines

def process_images(image_dir, output_dir):
    all_good_lines = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_image = {}
        for image in os.listdir(image_dir):
            if not image.endswith('.jpg'):
                continue
            image_path = os.path.join(image_dir, image)
            output_path = os.path.join(output_dir, image)
            future = executor.submit(process_image, image_path, output_path)
            future_to_image[future] = image

        for future in concurrent.futures.as_completed(future_to_image):
            image = future_to_image[future]
            try:
                good_lines = future.result()
                if good_lines:
                        all_good_lines.extend(good_lines)
            except Exception as exc:
                print(f'{image} generated an exception: {exc}')

    return all_good_lines

def extrude_line(good_lines):
    if not good_lines:
        print("No good lines")
    else:
        line = good_lines[8]
        x1, y1, x2, y2 = line
        print("Line: ", line)
        line2 = x1, y1-1000, x2, y2-1000
        print("Line2: ", line2)

        return line, line2


def draw_lines(l1, l2):

    img = cv.imread('../data/images/Horizontal/0b7d98ea-4200080346_5543_156.jpg')
    cv.line(img, (l1[0], l1[1]), (l1[2], l1[3]), (0, 255, 0), 2)
    cv.line(img, (l2[0], l2[1]), (l2[2], l2[3]), (0, 255, 0), 2)

    cv.imwrite('../data/images/0b7d98ea-4200080346_5543_156.jpg', img)

    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    image_dir = "../data/images/Horizontal/"
    output_dir = '../data/images/Horizontal_houghlines_images/'
    all_good_lines = process_images(image_dir, output_dir)
    # print("All good lines:", all_good_lines)
    l1, l2 = extrude_line(all_good_lines)
    draw_lines(l1, l2)

if __name__ == "__main__":
    main()