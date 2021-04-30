import logging
import logging.handlers
import queue
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from pathlib import Path
from typing import List, NamedTuple
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


st.set_page_config(page_title='Face Mask Detector', page_icon="./logo/logo.png" , layout='centered', initial_sidebar_state='expanded')
# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)


def main():
    st.header("Face Mask Detector 😷")

    object_detection_page = "Real time face mask detection"
    image_detection = (
        "Face mask detection on image"
    )
    
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            object_detection_page,
            image_detection,
        ],
    )
    st.subheader(app_mode)

    if app_mode == object_detection_page:
        app_object_detection()
    elif app_mode==image_detection:
        mask_detection()


def app_object_detection():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """

    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
   # print("[INFO] loading face mask detector model...")
    maskNet = load_model("face_mask_detector.model")
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoTransformer(VideoTransformerBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"
            

        def __init__(self) -> None:
            self._net=cv2.dnn.readNet(prototxtPath, weightsPath)
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()
        def _annotate_image(self, image, detections):
            
           
                # loop over the detection

        # grab the dimensions of the frame and then construct a blob
        # from it
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

                # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()

                # initialize our list of faces, their corresponding locations,
                # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

                # loop over the detections
            for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the detection
                confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the confidence is
                    # greater than the minimum confidence
                if confidence > self.confidence_threshold:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")


                        # ensure the bounding boxes fall within the dimensions of
                        # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                        # extract the face ROI, convert it from BGR to RGB channel
                        # ordering, resize it to 224x224, and preprocess it
                    face = image[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                        # add the face and bounding boxes to their respective
                        # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

                # only make a predictions if at least one face was detected
            if len(faces) > 0:
                    # for faster inference we'll make batch predictions on *all*
                    # faces at the same time rather than one-by-one predictions
                    # in the above `for` loop
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)

                # return a 2-tuple of the face locations and their corresponding
                # locations
            

      

                # loop over the detected face locations and their corresponding
                # locations
            for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                result.append(Detection(name=label, prob=float(confidence)))

                    # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            return image,result

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
                )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

                # NOTE: This `transform` method is called in another thread,
                # so it must be thread-safe.
            self.result_queue.put(result)

            return annotated_image

    #st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    #st.markdown('<h2 align="center">Real time detection on webcam</h2>', unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=MobileNetSSDVideoTransformer,
        async_transform=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold



def mask_image():
    
    global RGB_img
    # load our serialized face detector model from disk
    #print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
   # print("[INFO] loading face mask detector model...")
    model = load_model("face_mask_detector.model")

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    #print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_image()

def mask_detection():
    #st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    #st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
    st.markdown("### Upload your image here ⬇")
    image_file = st.file_uploader("", type=['jpg','png','jpeg'])  # upload image
    if image_file is not None:
        our_image = Image.open(image_file)  # making compatible to PIL
        im = our_image.save('./images/out.jpg')
        saved_image = st.image(image_file, caption='', use_column_width=True)
        st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
        if st.button('Process'):
            st.image(RGB_img, use_column_width=True)




if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
