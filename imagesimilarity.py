import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

threshold=0.7

def has_transparency(img):
    # Check if the image has an alpha channel
    return img.mode == 'RGBA' and 'A' in img.getbands()

def change_background_to_white(input_path, output_path):
    # Open the image
    img = Image.open(input_path)

    # Check if the image has transparency
    if has_transparency(img):
        # Create a new image with a white background
        new_img = Image.new("RGB", img.size, "white")

        # Paste the original image onto the new image, preserving transparency
        new_img.paste(img, (0, 0), img)

        # Save the result
        new_img.save(output_path)
    else:
        # If there is no transparency, simply save the original image
        img.save(output_path)

# treshold tuh kaya persentase, jadi imagenya dianggap similar kalau persentase kesamaannya diatas 70% (treshold 0.7 = 70%)
def image_similarity(image_path1, image_path2):
    output_image_path = "similarimage.jpg"   
    change_background_to_white(image_path2, output_image_path)

    # Load images
    real_image = cv2.imread(image_path1)
    similar_image = cv2.imread(image_path2)

    # Convert the training image to RGB
    real_rgb = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
    similar_rgb = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)
    
    #grayscale
    real_gray = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY)
    similar_gray = cv2.cvtColor(similar_rgb, cv2.COLOR_RGB2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    real_keypoints, real_descriptors = orb.detectAndCompute(real_gray, None)
    similar_keypoints, similar_descriptors = orb.detectAndCompute(similar_gray, None)

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match descriptors
    matches = bf.match(real_descriptors, similar_descriptors)

    # Sort them in ascending order of distance
    matches = sorted(matches, key = lambda x : x.distance)

    if(len(matches)>0):
        # Calculate similarity score
        similarity = sum([match.distance for match in matches]) / len(matches)
    else:
        similarity = 0
    
    result = cv2.drawMatches(real_rgb, real_keypoints, similar_gray, similar_keypoints, matches[:10], similar_gray, flags = 2)

    return similarity
    # # Resize the image
    # resized_image1 = cv2.resize(image1, (150, 80))
    # resized_image2 = cv2.resize(image2, (150, 80))

    # # Calculate Structural Similarity Index
    # similarity = ssim(resized_image1, resized_image2)                                                                 

    # if similarity > threshold:
    #     return similarity
    #     # return "The images are similar."
    # else:
    #     return similarity
        # return "The images are not similar."
