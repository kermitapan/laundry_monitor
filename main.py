import cv2
import json
from utils import extract_text_from_roi, is_clothes_present

# Load image
image_path = "images/WIN_20250522_22_21_29_Pro.jpg"
img = cv2.imread(image_path)

# Load ROI config
with open("rois_config.json", "r") as f:
    roi_config = json.load(f)

for machine_name, rois in roi_config.items():
    screen_roi = img[rois["screen"][1]:rois["screen"][3], rois["screen"][0]:rois["screen"][2]]
    drum_roi = img[rois["drum"][1]:rois["drum"][3], rois["drum"][0]:rois["drum"][2]]

    # Display the screen ROI
    cv2.imshow(f"Screen ROI - {machine_name}", screen_roi)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    time_text = extract_text_from_roi(screen_roi)
    clothes_status = is_clothes_present(drum_roi)

    print(f"Machine: {machine_name}")
    print(f"  Time Remaining: {time_text}")
    print(f"  Clothes Detected: {'Yes' if clothes_status else 'No'}")

