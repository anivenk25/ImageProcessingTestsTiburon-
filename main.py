import cv2
import numpy as np

def reduce_underwater_glare(image_path, gamma=1.2, clip_limit=2.0):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to float32 for gamma correction
    img_float = img.astype(np.float32) / 255.0

    # Apply gamma correction to reduce glare
    img_corrected = np.power(img_float, gamma)

    # Convert the corrected image back to uint8
    img_corrected = (img_corrected * 255).astype(np.uint8)

    # Convert to LAB color space
    lab_img = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2Lab)

    # Apply adaptive histogram equalization to the L channel
    lab_img[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lab_img[:, :, 0])

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    return enhanced_img

# Example usage
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"
output_image = reduce_underwater_glare(input_image_path)

# Display the original and processed images
cv2.imshow('Original Image', cv2.imread(input_image_path))
cv2.imshow('Processed Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
