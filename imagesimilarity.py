import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

output_image_path = "similarimage.jpg" 

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

def crop_image(input_path, output_path, left, top, right, bottom):
    # Open the image file
    img = Image.open(input_path)

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_img.save(output_path)

# treshold tuh kaya persentase, jadi imagenya dianggap similar kalau persentase kesamaannya diatas 70% (treshold 0.7 = 70%)
def image_similarity(image_real_path, image_similar_path):
    crop_image(image_similar_path, output_image_path, 51, 25, 399, 225)
    change_background_to_white(image_similar_path, output_image_path)

    # Load images
    real_image = cv2.imread(image_real_path)
    similar_image_source = cv2.imread(image_similar_path)
    
     # Resize the image
    resized_real = cv2.resize(real_image, (348, 200))
    resized_similar = cv2.resize(similar_image_source, (348, 200))
    
    # Convert the training image to Grayscale
    real_gray = cv2.cvtColor(resized_real, cv2.COLOR_BGR2GRAY)
    similar_gray = cv2.cvtColor(resized_similar, cv2.COLOR_BGR2GRAY)
    
    # Calculate Structural Similarity Index
    similarity = ssim(real_gray, similar_gray)
    
    return similarity
