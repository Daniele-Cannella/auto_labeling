import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def detect_and_dewarp(image_file, color_lower, color_upper, offset_y=900, resize_dim=(800, 600)):
    img = cv.imread('fe57de1d-4200079871_0819_156.jpg')

    if img is None:
        print(f"Error: Unable to load image {image_file}")
        return

    # Define the specific ROI coordinates
    x_start, y_start = 1000, 400
    x_end, y_end = 1900, 1800

    # Ensure ROI stays within image bounds
    height, width = img.shape[:2]
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    y_start = max(0, y_start)
    y_end = min(height, y_end)

    # Extract ROI
    roi = img[y_start:y_end, x_start:x_end]

    # Convert ROI to RGB
    roi_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)

    # Create mask based on color range
    mask = cv.inRange(roi_rgb, color_lower, color_upper)

    # Apply morphological operations to clean up the mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Find contours in the masked image
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found matching the specified color range.")
        return

    # Initialize points
    lowest_left = None
    lowest_right = None

    for contour in contours:
        for point in contour:
            x, y = point[0]
            # Ensure point is within ROI for more accurate detection
            if x < roi.shape[1] // 2:
                if lowest_left is None or y > lowest_left[1] or (y == lowest_left[1] and x < lowest_left[0]):
                    lowest_left = (x, y)
            else:
                if lowest_right is None or y > lowest_right[1] or (y == lowest_right[1] and x > lowest_right[0]):
                    lowest_right = (x, y)

    if lowest_left is None or lowest_right is None:
        print("No valid points found within the specified ROI.")
        return
    # Adjust points to the original image coordinates
    lowest_left = (lowest_left[0] + x_start - 300, lowest_left[1] + y_start -80)
    lowest_right = (lowest_right[0] + x_start, lowest_right[1] + y_start)

    # Calculate the top points by subtracting the offset in the y-axis
    top_left = (lowest_left[0], max(0, lowest_left[1] - offset_y))
    top_right = (lowest_right[0], max(0, lowest_right[1] - offset_y))

    # Ensure points are correctly ordered
    if top_left[1] > top_right[1]:
        top_left, top_right = top_right, top_left

    # Calculate width and height for the dewarped image
    dewarp_width = int(np.linalg.norm(np.array(lowest_right) - np.array(lowest_left)))
    dewarp_height = max(abs(lowest_left[1] - top_left[1]), abs(lowest_right[1] - top_right[1]))

    # Define source points for perspective transformation
    src_corners = np.float32([
        lowest_left,
        lowest_right,
        top_right,
        top_left
    ])

    # Define destination points with the same width and height
    dst_corners = np.float32([
        [0, dewarp_height],
        [dewarp_width, dewarp_height],
        [dewarp_width, 0],
        [0, 0]
    ])

    # Compute the perspective transform matrix
    M = cv.getPerspectiveTransform(src_corners, dst_corners)

    # Apply perspective transformation to dewarp the object
    dewarped = cv.warpPerspective(img, M, (dewarp_width, dewarp_height))

    # Optional: Resize the dewarped image
    dewarped_resized = cv.resize(dewarped, resize_dim)

    # Detect edges in the mask for Hough Lines
    edges = cv.Canny(mask, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw the lines on the dewarped image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(dewarped_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("No lines detected.")

    # Optional: Visualize the results
    plt.figure(figsize=(15, 5))
    plt.subplot(151), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Original Image')

    # Plot the detected points
    plt.scatter(*lowest_left, color='red')
    plt.scatter(*lowest_right, color='red')
    plt.scatter(*top_left, color='blue')
    plt.scatter(*top_right, color='blue')
    plt.subplot(152), plt.imshow(mask, cmap='gray'), plt.title('Color Mask')
    plt.subplot(153), plt.imshow(edges, cmap='gray'), plt.title('Edges for Hough')
    plt.subplot(154), plt.imshow(cv.cvtColor(dewarped, cv.COLOR_BGR2RGB)), plt.title('Dewarped Image')
    plt.subplot(155), plt.imshow(cv.cvtColor(dewarped_resized, cv.COLOR_BGR2RGB)), plt.title('Dewarped Image with Hough Lines')
    plt.tight_layout()
    plt.show()

    # Return dewarped image or perform further processing
    return dewarped_resized

# Example usage
image_path = '/mnt/data/fe57de1d-4200079871_0819_156.jpg'  # Replace with your image path
color_lower = np.array([113, 103, 104], dtype=np.uint8)  # Example lower bound for greenish color
color_upper = np.array([126, 117, 118], dtype=np.uint8)  # Example upper bound for greenish color
dewarped_image = detect_and_dewarp(image_path, color_lower, color_upper)

