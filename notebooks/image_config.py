import cv2
import matplotlib.pyplot as plt
import re
import numpy as np

kernel_size = np.ones((5, 5), np.uint8)

def convert_to_opencv_format(image_rgb_float):
    # Clip to [0, 1] range and convert to [0, 255]
    image_rgb_uint8 = np.clip(image_rgb_float*255, 0, 255).astype('uint8')
    # Convert RGB to BGR
    image_bgr_uint8 = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)
    return image_bgr_uint8

def gray_scaled_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarized_image(image):
    if len(image.shape) > 2:
        image = gray_scaled_image(image)
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded

def invert_image(image):
    return cv2.bitwise_not(image.copy())

def dilate_image(image): 
    image = binarized_image(image)
    return cv2.dilate(image, kernel_size, iterations=5)

def draw_contour_on_image(image,contour):
    img = image.copy()
    if len(contour) > 1:
        image_with_contours = cv2.drawContours(img, contour, -1, (255,255,0), 3)
    else:
        image_with_contours = cv2.drawContours(img, [contour], -1, (255,255,0), 3)
    image_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
    return image_with_contours
    
def find_and_draw_contours_in_image(image):
    dilated_image = dilate_image(image)
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = draw_contour_on_image(image,contours)
    return contours, image_with_contours

def filter_and_draw_rectangular_contours(image,contours,epsilon_factor = 0.01):
    rectangular_contours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        if len(approx) == 4:
            rectangular_contours.append(approx)
    image_with_only_rectangular_contours = image.copy()
    cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
    image_with_only_rectangular_contours = cv2.cvtColor(image_with_only_rectangular_contours, cv2.COLOR_BGR2RGB)
    return rectangular_contours , image_with_only_rectangular_contours

def find_largest_contour(image,rectangular_contours):
    max_area = 0
    contour_with_max_area = None
    for contour in rectangular_contours:
        print(contour)
        contour_area = cv2.contourArea(contour)
        if contour_area > max_area:
            max_area = contour_area 
            contour_with_max_area = contour
        print(f'max contour is {contour_with_max_area}')
        
    image_with_contour = draw_contour_on_image(image,contour_with_max_area)
    return image_with_contour, contour_with_max_area

def order_rectangle_points(points):
    # Identifying the top-left and bottom-right points
    top_left = min(points, key=lambda x: np.sum(x))
    bottom_right = max(points, key=lambda x: np.sum(x))

    # Removing identified points from the list to isolate the remaining two points
    remaining_points = [p for p in points if not (np.array_equal(p, top_left) or np.array_equal(p, bottom_right))]

    # Among the remaining points, identify top-right and bottom-left
    if remaining_points[0][0] > remaining_points[1][0]:
        top_right = remaining_points[0]
        bottom_left = remaining_points[1]
    else:
        top_right = remaining_points[1]
        bottom_left = remaining_points[0]

    # Ordered points: Top-left, Top-right, Bottom-right, Bottom-left
    ordered_points = [top_left, top_right, bottom_right, bottom_left]
    return ordered_points

def add_10_percent_padding(image,perspective_corrected_image):
    image_height = image.shape[0]
    padding = int(image_height * 0.1)
    perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def find_lines(img, threshold = 150, line_type='both'):
    # Load the image
    image = img.copy()
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Hough Transform to find lines
    lines = cv2.HoughLines(edges, 3, 3*(np.pi/180), threshold)  # Adjust '100' based on your needs

    # Check if any lines were found
    if lines is not None:
        # Filter and draw lines based on the specified type
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # For horizontal lines
            if line_type == 'horizontal' and (theta < np.pi/180 * 5 or theta > np.pi - np.pi/180 * 5):
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # For vertical lines
            elif line_type == 'vertical' and np.pi/4 <= theta <= 3*np.pi/4:
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # For both types
            elif line_type == 'both':
                if theta < np.pi/180 * 5 or theta > np.pi - np.pi/180 * 5:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif np.pi/4 <= theta <= 3*np.pi/4:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image


import cv2
import numpy as np

import numpy as np

def merge_close_lines(lines, delta_rho = 8, delta_theta = np.deg2rad(1)):
    grouped_lines = {}  # Dictionary to hold groups of lines
    line_indices = list(range(len(lines)))  # Keep track of line indices
    # Group lines by their distance (rho) and angle (theta)
    while line_indices:
        idx = line_indices.pop(0)
        rho, theta = lines[idx][0]
        group_key = None

        # Find a group for the current line
        for key in grouped_lines.keys():
            rho_group, theta_group = key
            if abs(rho - rho_group) < delta_rho and abs(theta - theta_group) < delta_theta:
                group_key = key
                break

        # Add line to the existing group or create a new group if none found
        if group_key is not None:
            grouped_lines[group_key].append((rho, theta))
        else:
            grouped_lines[(rho, theta)] = [(rho, theta)]

    # Average out the lines in each group to create a single line
    merged_lines = []
    for group in grouped_lines.values():
        rhos, thetas = zip(*group)
        avg_rho = np.mean(rhos)
        avg_theta = np.mean(thetas)
        merged_lines.append((avg_rho, avg_theta))

    return merged_lines


def erase_lines_from_binarized_image(img, line_type='both',threshold = 250):
    # Load the image
    image = img.copy()
    # Binarize the image, if not already
    _, binarized_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(binarized_image, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Hough Transform to find lines
    lines = cv2.HoughLines(edges, 1,np.pi/180, threshold)  # Adjust '100' based on your needs
    if lines is not None:
        merged_lines = merge_close_lines(lines, delta_rho = 8, delta_theta = np.deg2rad(1))
    else:
        merged_lines = []

    # Create a mask where lines will be drawn
    line_mask = np.zeros_like(binarized_image)

    # Check if any lines were found
    if lines is not None:
        # Filter and draw lines based on the specified type
        for rho, theta in merged_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # For horizontal lines
            if line_type == 'horizontal' and (theta < np.pi/180 * 5 or theta > np.pi - np.pi/180 * 5):
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            # For vertical lines
            elif line_type == 'vertical' and np.pi/4 <= theta <= 3*np.pi/4:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            # For both types
            elif line_type == 'both':
                if theta < np.pi/180 * 5 or theta > np.pi - np.pi/180 * 5:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                elif np.pi/4 <= theta <= 3*np.pi/4:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Remove the lines from the original binarized image by using the mask
    binarized_image[line_mask == 255] = 255

    return binarized_image


def erode_vertical_lines(inverted_image):
    hor = np.array([[1,1,1,1,1,1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=10)
    return vertical_lines_eroded_image

def erode_horizontal_lines(inverted_image):
    ver = np.array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=10)
    return horizontal_lines_eroded_image

def combine_eroded_images(vertical_lines_eroded_image,horizontal_lines_eroded_image):
    combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)
    return combined_image_dilated

def subtract_combined_and_dilated_image_from_original_image(inverted_image,combined_image_dilated):
    image_without_lines = cv2.subtract(inverted_image,combined_image_dilated)
    return image_without_lines

def remove_noisy_line_borders_from_image(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = img.copy()
    image = cv2.erode(image,kernel,iterations = 1)
    image = cv2.dilate(image,kernel,iterations = 1)
    return image
    