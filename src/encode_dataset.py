# python encode_dataset.py -i dataset -e encodings.pickle

import face_recognition
import argparse
import pickle
import os

# Build and parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# Initialize lists to store names and corresponding encoding vectors
knownEncodings = []
knownNames = []

# Iterate through directories and files in the dataset
for (dirpath, dirnames, filenames) in os.walk(args["dataset"]):
    for filename in filenames:
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Build the full path to the image file
            imagePath = os.path.join(dirpath, filename)
            print(f"[INFO] processing image: {imagePath}")

            # Get the name of the person from the directory name containing the image
            name = imagePath.split(os.path.sep)[-2]

            # Read and encode the image using the face_recognition library
            image = face_recognition.load_image_file(imagePath)
            encodings = face_recognition.face_encodings(image)

            # Add the name and encoding to the lists
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

# Serialize the list of encodings and names to a pickle file
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))
print("[INFO] encodings saved successfully")
