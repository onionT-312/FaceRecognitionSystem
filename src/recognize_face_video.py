# python recognize_face_video.py --encodings encodings.pickle

import face_recognition
import argparse
import pickle
import cv2

# Hàm callback để cập nhật kích thước khung hình khi thay đổi cửa sổ
def update_frame_size(event, x, y, flags, param):
    global video
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_width = 640  # Đặt chiều rộng của khung hình là 640 pixels
        frame_height = 480  # Đặt chiều cao của khung hình là 480 pixels
        video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Xây dựng và parse các tham số command line
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to output video")
args = vars(ap.parse_args())

# Load danh sách các khuôn mặt đã mã hóa và tên tương ứng
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Khởi tạo video stream và bộ ghi video nếu cần
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)
frame_width = 640  # Đặt chiều rộng của khung hình là 640 pixels
frame_height = 480  # Đặt chiều cao của khung hình là 480 pixels
video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Khởi tạo cửa sổ OpenCV và đặt callback cho việc cập nhật kích thước khung hình
cv2.namedWindow('Frame, press "q" to exit')
cv2.setMouseCallback('Frame, press "q" to exit', update_frame_size)

# Khởi tạo bộ ghi video nếu được chỉ định
writer = None

# Lặp qua từng frame từ video stream
while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    # Chuyển đổi từ BGR sang RGB (được yêu cầu bởi face_recognition)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện các khuôn mặt trong frame và mã hóa chúng
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Duyệt qua từng khuôn mặt được phát hiện trong frame
    for encoding, (top, right, bottom, left) in zip(encodings, boxes):
        # So sánh encoding của khuôn mặt với danh sách encoding đã biết
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # Kiểm tra xem encoding của khuôn mặt có khớp với danh sách encoding đã biết không
        if True in matches:
            # Lấy index của encoding đã khớp
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Đếm số lần xuất hiện của mỗi tên được nhận dạng
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Lấy tên có số lần xuất hiện nhiều nhất
            name = max(counts, key=counts.get)

        # Vẽ bounding box và hiển thị tên của người được nhận dạng lên frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Hiển thị frame có chứa các khuôn mặt và tên của người được nhận dạng
    cv2.imshow('Frame, press "q" to exit', frame)
    key = cv2.waitKey(1) & 0xFF

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if key == ord("q"):
        break

    # Ghi frame vào tệp video nếu có yêu cầu
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)  # Tăng FPS lên 30

    if writer is not None:
        writer.write(frame)

# Giải phóng bộ ghi video và đóng video stream
if writer is not None:
    writer.release()

video.release()
cv2.destroyAllWindows()
