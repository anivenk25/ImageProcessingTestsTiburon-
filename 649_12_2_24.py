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

def guided_filter(image, radius=15, epsilon=0.1):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply guided filter
    guided_filtered_img = cv2.ximgproc.guidedFilter(image, img_gray, radius, epsilon)

    # Scale the image intensities back to [0, 255]
    guided_filtered_img = (guided_filtered_img * 255).astype(np.uint8)

    return guided_filtered_img

# Example usage
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"

# Reduce glare
glare_reduced_img = reduce_underwater_glare(input_image_path, gamma=1.5)

# Apply guided filtering for clarity improvement
improved_img = guided_filter(glare_reduced_img, radius=15, epsilon=0.1)

# Display the original and enhanced images
cv2.imshow('Original Image', cv2.imread(input_image_path))
cv2.imshow('Enhanced Image', improved_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
