# python build_dataset.py

import cv2
import os

# Function to create a new directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to capture and save user images
def save_image(directory, img_counter, user_name, user_id):
    user_directory = os.path.join(directory, f"{user_name}_{user_id}")
    create_directory(user_directory)
    ret, frame = cap.read()
    img_name = f"{user_directory}/image_{img_counter}.png"
    cv2.imwrite(img_name, frame)
    print(f"{img_name} has been saved!")
    return img_counter + 1

# Create the dataset directory
dataset_path = "dataset"
create_directory(dataset_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Input the user's name
user_name = input("Enter your name: ")
user_id = input("Enter your ID: ")

# Create a directory for the user
user_directory = os.path.join(dataset_path, f"{user_name}_{user_id}")
create_directory(user_directory)

img_counter = 0  # Define and initialize the img_counter variable

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow('Webcam, press "c" to capture, "q" to exit', frame)

    key = cv2.waitKey(1)

    # Press "c" to save the image
    if key == ord('c'):
        img_counter = save_image(user_directory, img_counter, user_name, user_id)

    # Press "q" to exit
    elif key == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
