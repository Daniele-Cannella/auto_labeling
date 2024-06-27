import os
import cv2
import numpy as np

def get_images_path(path):
    jpg_images = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.jpg'):
            jpg_images.append(os.path.join(path, filename))
    return jpg_images

def find_intersection(line1, line2):
    """
    Find the intersection point of two lines given in Hesse normal form.
    """
    rho1, theta1 = line1[0], line1[1]
    rho2, theta2 = line2[0], line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    try:
        x0, y0 = np.linalg.solve(A, b)
        return [int(np.round(x0)), int(np.round(y0))]
    except np.linalg.LinAlgError:
        # This occurs when lines are parallel or coincident
        return None

def filter_points(points, x_range, y_range):
    """
    Filter points to be within the specified x and y range.
    """
    print("xrange = " + str(x_range))
    print("yrange = " + str(y_range))
    filtered_points = []
    for point in points:
        if 0 <= point[0] <= x_range and 0 <= point[1] <= y_range:
            filtered_points.append(point)
    return filtered_points


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def main():
    path = '/home/daniele/Downloads/tutto'
    images_path = get_images_path(path)

    # Define the ranges for the bottom points (adjust these ranges as needed)

    for image_path in images_path:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Unable to read image: {image_path}")
            continue

        # Define the coordinates based on your provided values
        bottom_right = (1650, 1850)
        bottom_left = (900, 1700)
        height = 1300;
        top_left = (bottom_left[0], bottom_left[1] - height)
        top_right = (bottom_right[0], bottom_right[1] - height)

        # Create a black background image of the same size as the original image
        black_bg = np.zeros_like(image)

        # Slice the image to keep only the region of interest
        sliced_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Calculate coordinates for placing the sliced region back into the black background
        x_offset = top_left[0]
        y_offset = top_left[1]

        # Place the sliced region into the black background image
        black_bg[y_offset:y_offset + sliced_region.shape[0],
        x_offset:x_offset + sliced_region.shape[1]] = sliced_region

        # Display or save the result
        #cv2.imshow('Result', black_bg)
        #cv2.waitKey(0)
        prova = black_bg.copy()
        hsv = cv2.cvtColor(prova, cv2.COLOR_BGR2HSV)
        print(f"hsv shape: {hsv.shape}")
        prova = cv2.resize(hsv, (400,400))
        #cv2.imshow("hsv", prova)
        #cv2.waitKey(0)

        # Define lower and upper bounds for brown color in HSV
        lower_brown = np.array([1, 10, 10])
        upper_brown = np.array([30, 120, 200])

        # Threshold the HSV image to get only brown colors
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_lines_image = cv2.bitwise_and(image, image, mask=mask)
        print(f"brown_lines shape: {brown_lines_image.shape}")
        #brown_lines_image_prova = cv2.resize(brown_lines_image, (400,400))
        #brown_lines_image = cv2.resize(brown_lines_image, (brown_lines_image.shape[0]//3, brown_lines_image.shape[1]//4))
        # cv2.imshow("brown_lines", brown_lines_image_prova)
        #cv2.waitKey(0)
        # Convert to grayscale and find edges
        brown_lines_image = cv2.cvtColor(brown_lines_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(brown_lines_image, 10, 20, apertureSize=5)
        edge_points = np.nonzero(edges)

        # Combine x and y coordinates into a list of [x, y] pairs
        edge_coordinates = np.column_stack((edge_points[1], edge_points[0]))

        # Convert to list of lists (if needed)
        intersections = edge_coordinates.tolist()
        print(edges)
        #prova_edges = edges.copy()
        #cv2.imshow("image", edges)

        if intersections is None:
            print(f"No lines found in {image_path}")
            continue

        print(f"Number of lines detected: {len(intersections)}")

        # Draw filtered lines on the image for visualization
        image_with_lines = image.copy()
        for intersection in intersections:
            x1, y1 = intersection
            cv2.circle(image_with_lines, (x1, y1), radius=4, color=(0, 0, 255), thickness=-1)


        print(f"Number of intersection points: {len(intersections)}")

        # Filter intersections to find points within the specified range
        bottom_points = filter_points(intersections, image_with_lines.shape[1], image_with_lines.shape[0])

        if len(bottom_points) < 2:
            print(f"Not enough bottom points found in {image_path}")
            continue

        # Sort the bottom points by x coordinate

        print(bottom_left)
        print(bottom_right)
        # Function to calculate Euclidean distance between two points

        # Initialize variables to store the closest points
        bottom_left_point = None
        bottom_right_point = None
        min_distance_to_bottom_left = float('inf')
        min_distance_to_bottom_right = float('inf')

        # Iterate through each point to find the closest to bottom-left and bottom-right vertices
        for point in intersections:
            # Calculate distance to bottom-left vertex
            dist_to_bottom_left = distance(point, bottom_left)
            if dist_to_bottom_left < min_distance_to_bottom_left:
                min_distance_to_bottom_left = dist_to_bottom_left
                bottom_left_point = point

            # Calculate distance to bottom-right vertex
            dist_to_bottom_right = distance(point, bottom_right)
            if dist_to_bottom_right < min_distance_to_bottom_right:
                min_distance_to_bottom_right = dist_to_bottom_right
                bottom_right_point = point

        # Calculate top points by adding height delta

        bottom_left = bottom_left_point
        bottom_right = bottom_right_point
        bottom_left[0] = bottom_left[0]-30
        bottom_right[0] = bottom_right[0] + 30
        print(f"Bottom left point: {bottom_left}")
        print(f"Bottom right point: {bottom_right}")
        top_left = (bottom_left[0]-100, bottom_left[1] - height)
        top_right = (bottom_right[0], bottom_right[1] - height)

        print(f"Top left point: {top_left}")
        print(f"Top right point: {top_right}")

        src_points = np.float32([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ])
        cv2.circle(image_with_lines, top_left, radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_lines, top_right, radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_lines, bottom_right, radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(image_with_lines, bottom_left, radius=10, color=(0, 255, 0), thickness=-1)

        # Define destination points
        dst_points = np.float32([
            [0, 0],     # Top-left corner
            [400, 0],   # Top-right corner
            [400, 400], # Bottom-right corner
            [0, 400]    # Bottom-left corner
        ])

        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Perform the perspective transformation
        warped_image = cv2.warpPerspective(image, M, (400, 400))

        resized_image_with_lines = cv2.resize(image_with_lines, (720, 480))

        # Display the original image with lines and warped image
        cv2.imshow('Original Image with Lines', resized_image_with_lines)
        cv2.imshow('Warped Image', warped_image)

        print("Displaying images. Press any key to continue to the next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
