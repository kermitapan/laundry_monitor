
import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'])

def extract_text_from_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    if results:
        return results[0][1]
    return "N/A"

def is_clothes_present(roi):
    # Simple heuristic: more texture or dark pixels may indicate clothes
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > 100  # Threshold — can be adjusted
