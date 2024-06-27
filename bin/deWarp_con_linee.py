import os
import cv2
import numpy as np
import random


def get_images_path(path):
    jpg_images = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.jpg'):
            jpg_images.append(os.path.join(path, filename))
    return jpg_images


def main():
    path = '/home/daniele/Downloads/tutto'
    images_path = get_images_path(path)

    bottom_right = (1650, 1850)
    bottom_left = (900, 1700)
    height = 1300;
    top_left = (bottom_left[0], bottom_left[1] - height)
    top_right = (bottom_right[0], bottom_right[1] - height)

    for image_path in images_path:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Unable to read image: {image_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply custom kernels to detect specific directions
        kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

        # Apply kernels to the region of interest
        filtered = cv2.filter2D(gray_image, -1, kernel)

        #cv2.imshow("Filtered", cv2.resize(filtered, (400, 400)))
        #cv2.waitKey(0)

        # Use Canny to detect edges
        #edges = cv2.Canny(np.uint8(combined_filtered), 50, 150)
        #cv2.imshow("Canny Edges", cv2.resize(edges, (400, 400)))
        #cv2.waitKey(0)




        black_bg = np.zeros_like(filtered)

        # Slice the image to keep only the region of interest
        sliced_region = filtered[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Calculate coordinates for placing the sliced region back into the black background
        x_offset = top_left[0]
        y_offset = top_left[1]

        # Place the sliced region into the black background image
        black_bg[y_offset:y_offset + sliced_region.shape[0],
        x_offset:x_offset + sliced_region.shape[1]] = sliced_region

        roi_image = black_bg
        #cv2.imshow("black_bg", cv2.resize(black_bg, (400, 400)))
        #cv2.waitKey(0)

        # Use Hough transform to detect lines
        lines = cv2.HoughLinesP(roi_image, 1, np.pi / 180, threshold=100, minLineLength=250, maxLineGap=10)

        # Draw lines on the original image
        x_range = (900, 1650)
        y_range = (1700, 1850)

        filtered_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if both endpoints are within the specified ranges
                if (x1 >= x_range[0] and x1 <= x_range[1] and
                        y1 >= y_range[0] and y1 <= y_range[1] and
                        x2 >= x_range[0] and x2 <= x_range[1] and
                        y2 >= y_range[0] and y2 <= y_range[1]):

                    if x2 - x1 != 0:  # Avoid division by zero
                        slope = (y2 - y1) / (x2 - x1)

                        # Draw line if slope is within the specified range
                        if slope > 0.1 and slope < 0.3:
                            filtered_lines.append(line)
                            #cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color
        # Calculate mean points for "left points" and "right points"
        left_x_sum = 0
        left_y_sum = 0
        right_x_sum = 0
        right_y_sum = 0
        count_left = 0
        count_right = 0

        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            if x1 < x2:  # Line is oriented left to right
                left_x_sum += x1
                left_y_sum += y1
                right_x_sum += x2
                right_y_sum += y2
                count_left += 1
                count_right += 1
            else:  # Line is oriented right to left
                left_x_sum += x2
                left_y_sum += y2
                right_x_sum += x1
                right_y_sum += y1
                count_left += 1
                count_right += 1

        # Calculate mean points
        if count_left > 0:
            left_mean_x = int(left_x_sum / count_left)
            left_mean_y = int(left_y_sum / count_left)
        else:
            left_mean_x = None
            left_mean_y = None

        if count_right > 0:
            right_mean_x = int(right_x_sum / count_right)
            right_mean_y = int(right_y_sum / count_right)
        else:
            right_mean_x = None
            right_mean_y = None

        # Draw the line passing through the mean points
        if left_mean_x is not None and left_mean_y is not None and right_mean_x is not None and right_mean_y is not None:
            filtered_point = ( (left_mean_x, left_mean_y), (right_mean_x, right_mean_y))
            left_mean_x, left_mean_y = filtered_point[0]
            right_mean_x, right_mean_y = filtered_point[1]

            # Draw circles on the image
            cv2.circle(image, (left_mean_x, left_mean_y), radius=20, color=(255, 0, 0),
                       thickness=-1)  # Blue circle at left_mean
            cv2.circle(image, (right_mean_x, right_mean_y), radius=20, color=(0, 0, 255), thickness=-1)

        point1 = filtered_point[0]
        point2 = filtered_point[1]
        x_min, x_max = 850, 1675
        x1, y1 = point1
        x2, y2 = point2

        # Calculate slope and intercept of the line passing through point1 and point2
        if x2 - x1 != 0:  # Avoid division by zero
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            # Calculate y-values at x_min and x_max
            y_at_x_min = int(m * x_min + c)
            y_at_x_max = int(m * x_max + c)

            final_point = ((x_min, y_at_x_min), (x_max, y_at_x_max))
        bottom_right = final_point[1]
        bottom_left = final_point[0]
        height = 1300
        top_left = (bottom_left[0], bottom_left[1] - height)
        top_right = (bottom_right[0], bottom_right[1] - height)

        src_points = np.float32([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ])
        image_with_lines = image.copy()
        cv2.circle(image_with_lines, top_left, radius=4, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_lines, top_right, radius=4, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_lines, bottom_right, radius=4, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_lines, bottom_left, radius=4, color=(0, 255, 0), thickness=-1)
        cv2.line(image_with_lines, bottom_left, top_left, color=(255, 0, 0), thickness=4)
        cv2.line(image_with_lines, top_left, top_right, color=(255, 0, 0), thickness=4)
        cv2.line(image_with_lines, top_right, bottom_right, color=(255, 0, 0), thickness=4)
        cv2.line(image_with_lines, bottom_right, bottom_left, color=(255, 0, 0), thickness=4)

        # Define destination points
        dst_points = np.float32([
            [0, 0],  # Top-left corner
            [400, 0],  # Top-right corner
            [400, 400],  # Bottom-right corner
            [0, 400]  # Bottom-left corner
        ])

        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Perform the perspective transformation
        warped_image = cv2.warpPerspective(image, M, (400, 400))

        resized_image_with_lines = cv2.resize(image_with_lines, (720, 480))

        # Display the original image with lines and warped image
        cv2.imshow('Original Image with Lines', resized_image_with_lines)
        cv2.imshow('Warped Image', warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
