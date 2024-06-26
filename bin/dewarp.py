'''
Operation of dewarping the image
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from icecream import ic


'''
from __future__ import print_function
import cv2
import numpy as np
 
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
 
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
 
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
 
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
 
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
 
  return im1Reg, h
 
if __name__ == '__main__':
 
  # Read reference image
  refFilename = "form.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
 
  # Read image to be aligned
  imFilename = "scanned-form.jpg"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
 
  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg, h = alignImages(im, imReference)
 
  # Write aligned image to disk.
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)
 
  # Print estimated homography
  print("Estimated homography : \n",  h)

'''
'''
def dewarp_image(image_path: str, output_path: str):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    cv.imwrite(output_path, rotated)
    return

def dewarp_images(image_dir: str, output_dir: str):
    for image in os.listdir(image_dir):
        if not image.endswith('.jpg'):
            continue
        print("Processing image: ", image)
        image_path = os.path.join(image_dir, image)
        output_path = os.path.join(output_dir, image)
        dewarp_image(image_path, output_path)
    return

if __name__ == "__main__":
    dewarp_images('../data/images/Horizontal/', '../data/images/Horizontal_dewarped_images/')
'''


def dewarp(image_path: str, output_path: str):
    img = cv.imread(image_path)

    ps1 = np.float32([[847, 720], [1680, 700], [920, 1610], [1650, 1720]])
    ps2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv.getPerspectiveTransform(ps1, ps2)

    dst = cv.warpPerspective(img, M, (300, 300))

    cv.imwrite(output_path, dst)
    # cv.imshow(output_path, dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return

def dewarp_images(image_dir: str, output_dir: str):
    print(image_dir, output_dir)
    for image in os.listdir(image_dir):
        if not image.endswith('.jpg'):
            continue
        # cv.imshow("Image", cv.imread(os.path.join(image_dir, image)))
        # cv.waitKey(0)
        print("Processing image: ", image)
        image_path = os.path.join(image_dir, image)
        output_path = os.path.join(output_dir, image)
        dewarp(image_path, output_path)
        # break
    return

if __name__ == "__main__":
    dewarp_images('../data/images/Horizontal/', '../data/images/Horizontal_dewarped_images/')

