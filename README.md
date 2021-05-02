# Face Mask Detection with Tensorflow, Keras and OpenCV
## About this project
Face Mask Detection system built with Tensorflow, keras and OpenCV using Deep Learning and Computer Vision concepts in order to detect face masks in static images, real-time video streams and in webapp

# USAGE
    Clone the repo

$ git clone https://github.com/Thioumbour/Face-Mask-Detection-with-Tensorflow-Keras.git

    Change your directory to the cloned repo

$ cd Face-Mask-Detection-with-Tensorflow-keras

    Create a Python virtual environment named 'test' and activate it

$ virtualenv test

$ source test/bin/activate

    Now, run the following command in your Terminal/Command Prompt to install the libraries required

$ pip3 install -r Requirements.txt

# WORKING

    Open terminal. Go into the cloned project directory and type the following command:

$ python3 train_mask_detector.py --dataset dataset

    To detect face masks in an image type the following command:

$ python3 detect_mask_image.py --image exemples/c.jpg

    To detect face masks in real-time video streams type the following command:

$ python3 detect_mask_video.py 

# Streamlit app

Face Mask Detector webapp using Tensorflow & Streamlit

command

$ streamlit run app.py 

# RESULTS

 Real time face mask detection on webcam
 
 
![vid2](https://user-images.githubusercontent.com/54810377/116796568-b5d40f00-aadd-11eb-8a88-60be0ebc896f.png)


Face mask detection on image


![out5](https://user-images.githubusercontent.com/54810377/116796585-e025cc80-aadd-11eb-8865-abc4f45a4356.png)
