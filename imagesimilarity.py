import cv2
from PIL import Image

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
    output_image_path = "similarimage.jpg"   
    change_background_to_white(image_similar_path, output_image_path)
    crop_image(image_similar_path, output_image_path, 128, 65, 322, 185)

    # Load images
    real_image = cv2.imread(image_real_path)
    similar_image_source = cv2.imread(image_similar_path)
    similar_image = cv2.resize(similar_image_source, (174, 100)) 
    
    # Convert the training image to Grayscale
    real_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    similar_gray = cv2.cvtColor(similar_image, cv2.COLOR_BGR2GRAY)
    
    # Convert the training image to Binary
    ret, real_gray = cv2.threshold(real_gray, 120, 255, cv2.THRESH_BINARY)
    ret, similar_gray = cv2.threshold(similar_gray, 120, 255, cv2.THRESH_BINARY)
    
    counter = 0
    for i in range(len(real_image)):
        for j in range(len(real_image[0])):
            # 0 black 255 white
            if real_gray[i][j] == similar_gray[i][j]:
                counter += 1

    similarity = counter / 17400
   
    return similarity
