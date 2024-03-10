import cv2
import numpy as np

def reduce_underwater_glare(image_path, gamma=1.5):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to float32
    img_float = img.astype(np.float32) / 255.0

    # Apply gamma correction to reduce glare
    img_corrected = np.power(img_float, gamma)

    # Convert the corrected image back to uint8
    img_corrected = (img_corrected * 255).astype(np.uint8)

    return img_corrected

def find_glare_pixels(image, window_size=5, threshold=0):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median filtering using a sliding window
    median_filtered = cv2.medianBlur(gray, window_size)

    # Find pixels with intensity above the threshold (potential glare pixels)
    glare_mask = (gray > threshold).astype(np.uint8) * 255

    # Replace potential glare pixels with the maximum value from the median filtered image
    glare_mask[glare_mask == 255] = median_filtered[glare_mask == 255].max()

    return glare_mask

# Example usage
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"

# Reduce glare
enhanced_img = reduce_underwater_glare(input_image_path, gamma=1.5)

# Find glare pixels and replace with median filtered values
glare_mask = find_glare_pixels(enhanced_img, window_size=5, threshold=200)

# Display the original and glare-reduced images
cv2.imshow('Original Image', cv2.imread(input_image_path))
cv2.imshow('Enhanced Image', enhanced_img)
cv2.imshow('Glare Pixel Replacement', glare_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
