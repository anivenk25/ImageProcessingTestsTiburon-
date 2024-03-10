import cv2
import numpy as np

class GlareFilter:
    def __init__(self, p=16, m=4):
        self.param = p
        self.min_param = m
        self.buff_param = []
        self.buff_min = []

    def median_filter(self, src):
        if len(self.buff_param) < self.param:
            self.buff_param.append(src.copy())
        else:
            self.buff_param.pop(0)
            self.buff_param.append(src.copy())
        avg_img = np.median(self.buff_param, axis=0).astype(np.uint8)
        return avg_img

    def min_median_filter(self, src):
        if len(self.buff_min) < self.min_param:
            self.buff_min.append(src.copy())
        else:
            self.buff_min.pop(0)
            self.buff_min.append(src.copy())
        min_img = self.buff_min[0]
        for i in range(1, len(self.buff_min)):
            min_img = np.minimum(min_img, self.buff_min[i])
        return self.median_filter(min_img)

def glare_reduction_and_contrast_enhancement(frame, glare_filter):
    # Convert frame to LAB color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Enhance contrast of the L channel using CLAHE
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))  # Increased clipLimit for richer colors
    l, a, b = cv2.split(lab_frame)
    l_enhanced = clahe.apply(l)

    # Merge enhanced L channel with original A and B channels
    enhanced_lab_frame = cv2.merge((l_enhanced, a, b))

    # Apply glare reduction filter separately to each color channel
    glare_reduced_frame = np.zeros_like(frame)
    for i in range(3):
        glare_reduced_frame[:, :, i] = glare_filter.min_median_filter(enhanced_lab_frame[:, :, i])

    # Enhance contrast of the glare-reduced frame using CLAHE again
    l_greduced, a_greduced, b_greduced = cv2.split(glare_reduced_frame)
    l_greduced_enhanced = clahe.apply(l_greduced)
    glare_reduced_enhanced_lab_frame = cv2.merge((l_greduced_enhanced, a_greduced, b_greduced))

    # Convert enhanced LAB frame back to BGR color space
    glare_reduced_enhanced_frame = cv2.cvtColor(glare_reduced_enhanced_lab_frame, cv2.COLOR_LAB2BGR)

    return glare_reduced_enhanced_frame


# Input and output video paths
input_video_path = r"E:\test2.mp4"
output_video_path = 'output_video.mp4'

# Open the input video file
input_video = cv2.VideoCapture(input_video_path)

# Get the video properties (width, height, and frame rate)
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(input_video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Initialize the glare filter
glare_filter = GlareFilter(p=5, m=3)

# Process each frame in the input video
while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Apply glare reduction and contrast enhancement to the frame
    processed_frame = glare_reduction_and_contrast_enhancement(frame, glare_filter)

    # Write the processed frame to the output video file after converting it to BGR color space
    output_video.write(processed_frame.astype(np.uint8))

# Release the video capture and writer objects
input_video.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
