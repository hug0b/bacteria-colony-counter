import cv2
import numpy as np


class BacteriaCounterProcessor:
    def __init__(self):
        pass

    def _perform_count(self, image_path: str, min_area: int, max_area: int, blur_kernel_size: int,
                       adaptive_threshold_block_size: int, adaptive_threshold_C: int,
                       morph_kernel_size: int, morph_iterations: int,
                       distance_transform_threshold: float, distance_transform_mask_size: int):
        # Read the image
        image = cv2.imread(image_path)

        # Resize the image while preserving the aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = width / height
        new_width = 1280
        new_height = int(new_width / aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray, (blur_kernel_size, blur_kernel_size), 0)

        # Apply adaptive thresholding to create a binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_threshold_block_size, adaptive_threshold_C)

        # Perform morphological operations to clean up the image
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        opening = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

        # Perform distance transform to separate close bacteria
        dist_transform = cv2.distanceTransform(
            opening, cv2.DIST_L2, distance_transform_mask_size)
        _, sure_fg = cv2.threshold(
            dist_transform, distance_transform_threshold * dist_transform.max(), 255, 0)

        # Find contours
        contours, _ = cv2.findContours(
            sure_fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        bacteria_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                bacteria_count += 1
                # Draw the contours on the image using the selected color and thickness
                cv2.drawContours(image, [contour], 0, (0, 255, 0), 1)

        return bacteria_count, image
