import cv2
import numpy as np


def remove_glare(image, threshold=220):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Compute luminance channel
    l_channel = lab_image[:, :, 0]

    # Thresholding to detect glare spots (adjust threshold value as needed)
    _, glare_mask = cv2.threshold(l_channel, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to enhance mask
    kernel = np.ones((5, 5), np.uint8)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of glare spots
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over detected glare spots
    for contour in contours:
        # Get contour area
        area = cv2.contourArea(contour)
        if area < 10:  # Skip small contours
            continue

        # Get bounding box around glare spot
        x, y, w, h = cv2.boundingRect(contour)

        # Create a mask for the current contour
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

        # Apply the mask to the region of interest
        roi = cv2.bitwise_and(image[y:y + h, x:x + w], mask[y:y + h, x:x + w])

        # Calculate median color within the region of interest
        median_color = np.median(roi, axis=(0, 1))

        # Replace pixels within glare spot with median color
        image[y:y + h, x:x + w] = median_color.astype(np.uint8)

    return image


# Create a CLAHE object (Arguments are optional and can be tuned)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Initialize variables to store the previous frames
prev_frames = []

# Number of frames to consider for median calculation
num_frames = 10

# Capture video from webcam or file
cap = cv2.VideoCapture(r"E:\test2.mp4")

if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Remove glare spots
    processed_frame = remove_glare(frame)

    # Convert the processed frame to grayscale
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to the grayscale frame
    clahe_frame = clahe.apply(gray_frame)

    # Convert back to BGR
    clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)

    # Show the processed frame
    cv2.imshow('Processed Frame', clahe_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
