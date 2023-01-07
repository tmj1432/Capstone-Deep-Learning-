import numpy as np
import cv2
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad'}
model =load_model('model3_phase1.h5')

#load face
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        inHeight = 300
        inWidth = 0
        frameOpenCVHaar = img.copy()
        frameHeight = frameOpenCVHaar.shape[0]
        frameWidth = frameOpenCVHaar.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
        frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(frameGray)
        bboxes = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                      int(x2 * scaleWidth), int(y2 * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)

        cropped_image = frameOpenCVHaar[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        final_image = cv2.resize(cropped_image, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)  ## need fourth dimension
        final_image = final_image / 255.0
        predictions = model.predict(final_image, verbose=0)
        predicted_class = np.argmax(predictions)
        if predicted_class == 0:
            status = 'Angry'
            cv2.putText(img, 'Please Send Assistance', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif predicted_class == 1:
            status = 'Happy'
        elif predicted_class == 2:
            status = 'Neutral'
        else:
            status = 'Sad'

        cv2.putText(img, f"Emotion: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return img


def main():
    # Face Analysis Application #
    st.title("Facial Emotion Recognition 😊😡😐😭")
    dropdown = ["Live Webcam","About"]
    choice = st.sidebar.selectbox("Select", dropdown)
    st.sidebar.markdown("By Tan Ming Jie")

    if choice == "Live Webcam":
        st.header("Webcam Live Feed")
        st.write("Click on start to detect your facial expression")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("Github Link")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    https://github.com/tmj1432/Human-Emotion-Recognition </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()