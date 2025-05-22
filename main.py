import cv2
import json
import numpy as np
import os
from utils import extract_text_from_roi, is_clothes_present

# Create the outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load image
image_path = "images/WIN_20250522_22_21_29_Pro.jpg"
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to highlight the white areas
_, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours to find bounding boxes
roi_coordinates = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y + h, x:x + w]  # Extract the ROI from the image

    # Use OCR to extract text from the ROI
    text = extract_text_from_roi(roi)

    # Check if the text contains numbers
    if any(char.isdigit() for char in text):
        roi_coordinates.append((x, y, x + w, y + h))
        # Draw the bounding box on the original image for visualization
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the image with bounding boxes for inspection
cv2.imwrite("outputs/roi_detected_with_numbers.jpg", img)
print("ROI coordinates with numbers detected and saved to outputs/roi_detected_with_numbers.jpg")

# Print the detected ROI coordinates
print("Detected ROI coordinates with numbers:")
for i, coords in enumerate(roi_coordinates):
    print(f"ROI {i + 1}: {coords}")

# Zoom into each detected ROI with numbers
for i, (x1, y1, x2, y2) in enumerate(roi_coordinates):
    roi = img[y1:y2, x1:x2]  # Extract the ROI

    # Apply a color mask to the ROI
    mask = np.zeros_like(roi)
    mask[:] = (0, 255, 0)  # Green color mask
    roi_with_mask = cv2.addWeighted(roi, 0.5, mask, 0.5, 0)

    # Zoom the ROI
    zoom_factor = 2
    zoomed_roi = cv2.resize(roi_with_mask, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    # Save the zoomed ROI
    zoomed_roi_path = f"outputs/zoomed_roi_{i + 1}.jpg"
    cv2.imwrite(zoomed_roi_path, zoomed_roi)
    print(f"Zoomed ROI {i + 1} saved to {zoomed_roi_path}")

