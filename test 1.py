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

    # Convert median_filtered to three channels to match image
    median_filtered_color = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)

    # Replace potential glare pixels with the corresponding median filtered values
    image_with_glare_replaced = np.copy(image)
    image_with_glare_replaced[glare_mask == 255] = median_filtered_color[glare_mask == 255]

    return image_with_glare_replaced

# Example usage
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"

# Reduce glare
enhanced_img = reduce_underwater_glare(input_image_path, gamma=1.5)

# Find glare pixels and replace with median filtered values
glare_replaced_img = find_glare_pixels(enhanced_img, window_size=5, threshold=200)

# Display the original and glare-reduced images
cv2.imshow('Original Image', cv2.imread(input_image_path))
cv2.imshow('Enhanced Image', enhanced_img)
cv2.imshow('Glare Pixel Replacement', glare_replaced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
