import cv2
import numpy as np

def segment_tumor(image_path):
    image = cv2.imread(image_path, 0)  # Load as grayscale
    _, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    image_color = cv2.imread(image_path)
    cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)
    
    cv2.imwrite("segmented_output.jpg", image_color)
    return "segmented_output.jpg"

# Test segmentation
segment_tumor("datasets/no/no8.jpg")
