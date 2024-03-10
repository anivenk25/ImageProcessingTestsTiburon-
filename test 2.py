import cv2
import numpy as np

def enhance_contrast(image, alpha=1.5, beta=30):
    # Apply contrast and brightness adjustment
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return enhanced_image

def reduce_underwater_glare(image_path, gamma=1.5, clip_limit=2.5):
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

def blend_with_background(enhanced_img, background_path, alpha=0.9):
    # Read the background image
    background = cv2.imread(background_path)

    # Resize the enhanced image to match the background size
    enhanced_img_resized = cv2.resize(enhanced_img, (background.shape[1], background.shape[0]))

    # Blend the enhanced image with the background
    blended_img = cv2.addWeighted(background, 1 - alpha, enhanced_img_resized, alpha, 0)

    return blended_img

# Example usage with adjusted parameters and increased blending and contrast
input_image_path = r"C:\Users\aniru\Downloads\bin_1.png"
background_path = r"C:\Users\aniru\Downloads\background.png"

# Reduce glare and enhance contrast
enhanced_img = reduce_underwater_glare(input_image_path, gamma=1.5, clip_limit=2.5)

# Enhance contrast further
enhanced_img = enhance_contrast(enhanced_img, alpha=2.5, beta=30)

# Blend with the background and increase blending
blended_image = blend_with_background(enhanced_img, background_path, alpha=0.2)

# Display the original, glare-reduced, contrast-enhanced, and blended images
cv2.imshow('Original Image', cv2.imread(input_image_path))
cv2.imshow('Enhanced Image', enhanced_img)
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
