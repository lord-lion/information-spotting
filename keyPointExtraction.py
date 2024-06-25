# # # SIFT (Scale-Invariant Feature Transform)

# # ## Import resources and display image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the SIFT detector object
sift = cv2.SIFT_create()

# Define the dataset and output directories
dataset_dir = './dataset/images/'
output_dir = './output/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all images in the dataset directory
image_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

# Load and process each image in the dataset
for image_file in image_files:
    # Load the image
    image_path = os.path.join(dataset_dir, image_file)
    image = cv2.imread(image_path)
    
    # Convert the image to RGB and gray scale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    
    # Draw keypoints without size and orientation
    keypoints_without_size = np.copy(image_rgb)
    cv2.drawKeypoints(image_rgb, keypoints, keypoints_without_size, color=(0, 255, 0))
    
    # Draw keypoints with size and orientation
    keypoints_with_size = np.copy(image_rgb)
    cv2.drawKeypoints(image_rgb, keypoints, keypoints_with_size, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Save the images with keypoints
    output_without_size_path = os.path.join(output_dir, f'keypoints_without_size_{image_file}')
    output_with_size_path = os.path.join(output_dir, f'keypoints_with_size_{image_file}')
    
    cv2.imwrite(output_without_size_path, cv2.cvtColor(keypoints_without_size, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_with_size_path, cv2.cvtColor(keypoints_with_size, cv2.COLOR_RGB2BGR))
    
    # Display keypoints
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))
    plots[0].set_title(f"Keypoints of {image_file} with size")
    plots[0].imshow(cv2.cvtColor(keypoints_with_size, cv2.COLOR_BGR2RGB))
    plots[1].set_title(f"Keypoints of {image_file} without size")
    plots[1].imshow(cv2.cvtColor(keypoints_without_size, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Print the number of keypoints detected
    print(f"Number of Keypoints Detected in {image_file}: ", len(keypoints))

# Optional: Matching keypoints between each image and a test image

# Load the test image
# test_image = cv2.imread('./dataset/images/0001.jpg')
# test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# test_image_gray = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2GRAY)

# # Create test image by adding Scale Invariance and Rotational Invariance
# test_image_down = cv2.pyrDown(test_image_rgb)
# test_image_down = cv2.pyrDown(test_image_down)
# num_rows, num_cols = test_image_down.shape[:2]

# rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
# test_image_rotated = cv2.warpAffine(test_image_down, rotation_matrix, (num_cols, num_rows))

# test_gray = cv2.cvtColor(test_image_rotated, cv2.COLOR_RGB2GRAY)

# # Detect keypoints and compute descriptors for the test image
# test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)

# for image_file in image_files:
#     # Load the image
#     image_path = os.path.join(dataset_dir, image_file)
#     image = cv2.imread(image_path)
    
#     # Convert the image to gray scale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Detect keypoints and compute descriptors
#     keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    
#     # Create a Brute Force Matcher object
#     bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    
#     # Perform the matching between the SIFT descriptors of the training image and the test image
#     matches = bf.match(descriptors, test_descriptor)
    
#     # The matches with shorter distance are the ones we want
#     matches = sorted(matches, key=lambda x: x.distance)
    
#     result = cv2.drawMatches(image_rgb, keypoints, test_image_rotated, test_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
#     # Save the matching result
#     output_match_path = os.path.join(output_dir, f'match_{image_file}')
#     cv2.imwrite(output_match_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
#     # Display the best matching points
#     plt.title('Best Matching Points')
#     plt.imshow(result)
#     plt.show()
    
#     # Print the total number of matching points
#     print(f"\nNumber of Matching Keypoints Between {image_file} and Test Image: ", len(matches))
