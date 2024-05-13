import numpy as np
import cv2  
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read("trainner.yml")                                                                     # helps to read the trained model

labels = {}
with open("labels.pickle", 'rb') as f:                                                              # loads the labels from the pickle file to the variables
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# the code below just opens the camera and detects the faces or roi
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]  # (ycord_start, ycord_end)
        roi_color = frame[y : y + h, x : x + w]
        # implementation of recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf>= 45 and conf<=85:                                                                  # if the prediction is more then 45% then print the label and name 
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX         
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)                    # this puts the text on top off the rectangle
        img_item = "my-images"
        # cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
