# python encode_dataset.py -i dataset -e encodings.pickle

import face_recognition
import argparse
import pickle
import os

# Xây dựng và parse các tham số command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# Khởi tạo danh sách tên và danh sách các vector encoding tương ứng
knownEncodings = []
knownNames = []

# Lặp qua các thư mục và tệp trong dataset
for (dirpath, dirnames, filenames) in os.walk(args["dataset"]):
    for filename in filenames:
        # Kiểm tra xem tệp có phải là ảnh không
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Xây dựng đường dẫn đầy đủ đến tệp ảnh
            imagePath = os.path.join(dirpath, filename)
            print(f"[INFO] processing image: {imagePath}")

            # Lấy tên của người từ tên thư mục chứa ảnh
            name = imagePath.split(os.path.sep)[-2]

            # Đọc và mã hóa ảnh bằng thư viện face_recognition
            image = face_recognition.load_image_file(imagePath)
            encodings = face_recognition.face_encodings(image)

            # Thêm tên và encoding vào danh sách
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

# Lưu danh sách encoding và tên vào tệp pickle
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] encodings saved successfully")