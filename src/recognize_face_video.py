# python recognize_face_video.py --encodings encodings.pickle

import face_recognition
import argparse
import pickle
import cv2

# Callback function to update frame size when resizing the window
def update_frame_size(event, x, y, flags, param):
    global video
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_width = 640  # Set the frame width to 640 pixels
        frame_height = 480  # Set the frame height to 480 pixels
        video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Build and parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to output video")
args = vars(ap.parse_args())

# Load the serialized facial encodings and corresponding names
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Start the video stream and initialize video writer if specified
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)
frame_width = 640  # Set the frame width to 640 pixels
frame_height = 480  # Set the frame height to 480 pixels
video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize the OpenCV window and set callback for updating frame size
cv2.namedWindow('Frame, press "q" to exit')
cv2.setMouseCallback('Frame, press "q" to exit', update_frame_size)

# Initialize video writer if specified
writer = None

# Loop over each frame from the video stream
while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    # Convert BGR to RGB (required by face_recognition)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame and compute their encodings
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over each detected face in the frame
    for encoding, (top, right, bottom, left) in zip(encodings, boxes):
        # Compare the encoding with known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # Check if there is a match
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Count the occurrences of each recognized name
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Get the name with the most occurrences
            name = max(counts, key=counts.get)

        # Draw bounding box and display the name of the recognized person
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frame with detected faces and names
    cv2.imshow('Frame, press "q" to exit', frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit loop when 'q' is pressed
    if key == ord("q"):
        break

    # Write frame to video file if specified
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)  # Increase FPS to 30

    if writer is not None:
        writer.write(frame)

# Release video writer and close video stream
if writer is not None:
    writer.release()

video.release()
cv2.destroyAllWindows()
