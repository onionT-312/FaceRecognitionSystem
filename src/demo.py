import tkinter as tk
from tkinter import filedialog
import os
import subprocess

def get_current_directory():
    return os.path.dirname(os.path.realpath(__file__))

def build_dataset():
    current_dir = get_current_directory()
    subprocess.run(["python", os.path.join(current_dir, "build_dataset.py")])
    
    encode_dataset()

def encode_dataset():
    current_dir = get_current_directory()
    subprocess.run(["python", os.path.join(current_dir, "encode_dataset.py"), "-i", current_dir, "-e", os.path.join(current_dir, "encodings.pickle")])

def recognize_faces():
    current_dir = get_current_directory()
    subprocess.run(["python", os.path.join(current_dir, "recognize_face_video.py"), "--encodings", os.path.join(current_dir, "encodings.pickle")])


root = tk.Tk()
root.title("Face Recognition System")

frame = tk.Frame(root, padx=20, pady=10)
frame.place(relx=0.5, rely=0.5, anchor="center")

build_button = tk.Button(frame, text="Tạo người dùng mới", command=build_dataset)
build_button.grid(row=0, column=0, padx=10, pady=5)

recognize_button = tk.Button(frame, text="Nhận diện người dùng", command=recognize_faces)
recognize_button.grid(row=0, column=1, padx=10, pady=5)

root.mainloop()
