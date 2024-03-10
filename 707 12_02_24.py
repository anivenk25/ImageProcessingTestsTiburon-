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

    # Convert the images to cv2.UMat objects
    image_umat = cv2.UMat(image)
    img_gray_umat = cv2.UMat(img_gray)

    # Apply guided filter
    guided_filtered_img = cv2.ximgproc.guidedFilter(image_umat, img_gray_umat, radius, epsilon)

    # Convert the filtered image back to a numpy.ndarray object
    guided_filtered_img = guided_filtered_img.get()

    # Scale the image intensities back to [0, 255]
    guided_filtered_img = (guided_filtered_img * 255).astype(np.uint8)

    return guided_filtered_img

# Example usage
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"

# Read the original image
original_image = cv2.imread(input_image_path)

# Reduce glare
glare_reduced_img = reduce_underwater_glare(input_image_path, gamma=1.5)

# Apply guided filtering for clarity improvement
improved_img = guided_filter(glare_reduced_img, radius=15, epsilon=0.1)

# Check if improved_img is a grayscale image
if len(improved_img.shape) == 2 or improved_img.shape[2] == 1:
    improved_img_gray = improved_img
else:
    improved_img_gray = cv2.cvtColor(improved_img, cv2.COLOR_BGR2GRAY)

# Create a mask for areas where the image is white
mask = cv2.inRange(improved_img_gray, 255, 255)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Get the part of the original image that corresponds to the white parts in the improved image
original_part = cv2.bitwise_and(original_image, original_image, mask=mask)

# Get the part of the improved image that is not white
improved_part = cv2.bitwise_and(improved_img, improved_img, mask=mask_inv)

# Resize improved_part to match the size of original_part
improved_part = cv2.resize(improved_part, (original_part.shape[1], original_part.shape[0]))

# If original_part is a grayscale image and improved_part is a color image, convert improved_part to grayscale
if len(original_part.shape) == 2 and len(improved_part.shape) == 3:
    improved_part = cv2.cvtColor(improved_part, cv2.COLOR_BGR2GRAY)

# If original_part is a color image and improved_part is a grayscale image, convert improved_part to color
elif len(original_part.shape) == 3 and len(improved_part.shape) == 2:
    improved_part = cv2.cvtColor(improved_part, cv2.COLOR_GRAY2BGR)

# Combine the original part and the improved part
final_image = cv2.add(original_part, improved_part)

# Convert the final image to grayscale if it is not already
if len(final_image.shape) == 3:
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a color image
final_image_color = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)


# Convert the grayscale image to a color image
final_image_color = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

# Create a mask for the white parts of the grayscale image
mask_white = cv2.inRange(final_image, 140, 255)

# Invert the mask
mask_white_inv = cv2.bitwise_not(mask_white)

# Get the part of the original image that corresponds to the white parts in the grayscale image
original_part_white = cv2.bitwise_and(original_image, original_image, mask=mask_white)

# Get the part of the color image that is not white
final_part_not_white = cv2.bitwise_and(final_image_color, final_image_color, mask=mask_white_inv)

# Combine the original part and the final part
final_image_combined = cv2.add(original_part_white, final_part_not_white)

# Display the original, improved, and final images
cv2.imshow('Original Image', original_image)
cv2.imshow('Improved Image', improved_img)
cv2.imshow('Final Image', final_image)
cv2.imshow('Combined Image', final_image_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()