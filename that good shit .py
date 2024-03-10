import cv2
import numpy as np

# integrate the sliding window median into this and make it work

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

def find_glare_pixels(image, threshold=0):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Print min and max values for debugging
    print("Min Value:", np.min(gray))
    print("Max Value:", np.max(gray))

    # Find pixels with intensity above the threshold (potential glare pixels)
    glare_mask = (gray > threshold).astype(np.uint8) * 255

    return glare_mask


def blend_with_background_and_glare(image, background_path, glare_mask, alpha=0.7):
    # Read the background image
    background = cv2.imread(background_path)

    # Resize the image and glare mask to match the background size
    image_resized = cv2.resize(image, (background.shape[1], background.shape[0]))
    glare_mask_resized = cv2.resize(glare_mask, (background.shape[1], background.shape[0]))

    # Invert the glare mask to obtain the non-glare mask
    non_glare_mask = cv2.bitwise_not(glare_mask_resized)

    # Blend the image with the background using the non-glare mask
    blended_img = cv2.addWeighted(background, 1 - alpha, image_resized, alpha, 0)

    # Replace the glare pixels from the original image onto the blended image
    blended_img_with_glare = cv2.bitwise_and(image_resized, image_resized, mask=glare_mask_resized)
    blended_img_with_glare = cv2.add(blended_img, blended_img_with_glare)

    return blended_img_with_glare


# Example usage
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"
background_path = r"C:\Users\aniru\Downloads\background.png"

# Reduce glare
enhanced_img = reduce_underwater_glare(input_image_path, gamma=1.5)

# Find glare pixels
glare_mask = find_glare_pixels(enhanced_img, threshold=200)

# Blend with background and emphasize glare pixels
result = blend_with_background_and_glare(enhanced_img, background_path, glare_mask, alpha=0.9)

# Display the original, glare-reduced, and blended images
cv2.imshow('Original Image', cv2.imread(input_image_path))
cv2.imshow('Enhanced Image', enhanced_img)
cv2.imshow('Blended Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

