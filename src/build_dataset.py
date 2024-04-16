# python build_dataset.py

import cv2
import os

# Hàm để tạo thư mục mới nếu nó chưa tồn tại
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Hàm chụp và lưu ảnh của người dùng
def save_image(directory, img_counter, user_name, user_id):
    user_directory = os.path.join(directory, f"{user_name}_{user_id}")
    create_directory(user_directory)
    ret, frame = cap.read()
    img_name = f"{user_directory}/image_{img_counter}.png"
    cv2.imwrite(img_name, frame)
    print(f"{img_name} đã được lưu!")
    return img_counter + 1

# Tạo thư mục dataset
dataset_path = "dataset"
create_directory(dataset_path)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Nhập tên từ người dùng
user_name = input("Nhập tên của bạn: ")
user_id = input("Nhập ID của bạn: ")

# Tạo thư mục cho người dùng
user_directory = os.path.join(dataset_path, f"{user_name}_{user_id}")
create_directory(user_directory)

img_counter = 0  # Định nghĩa và khởi tạo biến img_counter

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow('Webcam, press "c" to capture, "q" to exit', frame)

    key = cv2.waitKey(1)

    # Nhấn "s" để lưu ảnh
    if key == ord('c'):
        img_counter = save_image(user_directory, img_counter, user_name, user_id)

    # Nhấn "q" để thoát
    elif key == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()
