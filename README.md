
# Face Recognition System using Python

This is a live face recognition system via camera using OpenCV, Python, Deep Learning


## Installation

To run this project, you will need to install some additional libraries

* OpenCV
```bash
  pip install opencv-python
```
* Face Recognition
You can install Dlib (using python 3.7 -3.11): [click here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)

Next step: 

```bash
  pip install face_recognition
```
* Tkinter
```bash
  pip install tkintertable
```
* Pickle
```bash
  pip install pypickle
```
    
## General layout

The project will have several folders as follows

`build_dataset.py` - used to create a dataset

`encode_dataset.py` - encoding (128-d vector) for faces

`recognize_faces_video.py` - recognize faces from video webcams

`encodings.pickle` - encoding generated from `encode_faces.py` will be saved to disk information via this file

You can run the `demo.py` file to run the full program
```bash
    python demo.py
```

### Step 1: Create dataset
Here we use `build_dataset.py` to build the dataset. In the dataset folder there are subdirectories for each person with name + ID, each subdirectory contains a photo of that person's face.

You use the `c` key to take a photo, and the `q` key to quit.

Each person should take 15-20 photos for the model to be highly accurate

```bash
  python build_dataset.py
```
![image1](https://github.com/onionT-312/FaceRecognitionSystem/blob/main/picture/demo1.jpg)

### Step 2. Create encoding for faces in the dataset
After creating the dataset, we will create the encodings (or embeddings) of the faces in that dataset. The first thing to do is to extract the face ROIs (avoid using the entire image because there will be a lot of background noise affecting the model quality). To detect and extract faces, we can use many methods such as haar cascades, HOG + Linear SVM, Deep Learning-based face detector... Once we have face ROIs, we will pass them through the NN network to get the encodings.

![image2](https://github.com/onionT-312/FaceRecognitionSystem/blob/main/picture/demo2.jpg)

In this section the `encode_faces.py` file is run to save encodings and names + ID. The encodings and names are saved to the `encodings.pickle` file.
```bash
  python encode_dataset.py -i dataset -e encodings.pickle
```
### Step 3. Recognize faces in videos
Run the `recognize_faces_video.py` file to recognize faces in videos
![image3](https://github.com/onionT-312/FaceRecognitionSystem/blob/main/picture/demo3.jpg)
```bash
  python recognize_face_video.py --encodings encodings.pickle
```

## References
1. [https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213](https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213)
2. [https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
3. [https://github.com/huytranvan2010/Face-Recognition-with-OpenCV-Python-DL](https://github.com/huytranvan2010/Face-Recognition-with-OpenCV-Python-DL)
## Authors

[@onionT-312](https://github.com/onionT-312)

