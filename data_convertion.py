import os
import cv2
from PIL import Image                                                                           # pip install pillow --upgrade
import numpy as np
import pickle                                                                                   # to create a pickle file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))                                           # the main path in which this file is located
image_dir = os.path.join(BASE_DIR, "images")                                                    # path to the images folder
facefound = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')                        # a predefined model which detect faces
recognizer = cv2.face.LBPHFaceRecognizer_create()                                               # also a model which recognizes faces based on the images provided in the trainnig 
current_id = 0                                                                                  # ids which is used in the dictionary
labels_id = {}                                                                                  # labels dictionary
x_train = []                                                                                    # empty list that will contain the images in array format
y_labels = []                                                                                   # does the same job but with images

for root,dirs,files in os.walk(image_dir):                                                      # from line 16 to line 20 the code looks for a file with .png or .jpg and stores it in variables
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = (os.path.join(root, file))
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # print(label, path)
            if not label in labels_id:                                                          # if we have a new label then we add it to the dictionary
                labels_id[label] = current_id
                current_id +=1
            id_ = labels_id[label]
            # print(labels_id)
            pil_images = Image.open(path).convert("L")                                          # converts the images to grayscale
            image_array = np.array(pil_images, "uint8")                                         # converts the pil images to numpy
            # print(image_array)  
            faces = facefound.detectMultiScale(image_array,1.3, 5)  
            for (x,y,w,h) in faces:                                                             
                roi = image_array[y:y+h, x:x+w]                                                 # used to form the region of interst ie faces
                x_train.append(roi)                                                             # faces are then added to the empty list
                y_labels.append(id_)                                                            # labels are added to the empty list

# print(x_train)
# print(y_labels)

with open("labels.pickle", 'wb') as f:                                                          # forms a pickle file and dumps the lable ids in them
    pickle.dump(labels_id, f)

recognizer.train(x_train, np.array(y_labels))                                                    
recognizer.save("trainner.yml")                                                                 # creats a .yml trained model