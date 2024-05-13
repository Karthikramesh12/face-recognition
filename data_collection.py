import cv2                                                                              # a module that helps to open camera detect stuffs and etc
import os                                                                               # a module that helps to create new files and check is there is any file with same name and more

video = cv2.VideoCapture(0)                                                             # to open the camera            # haarcascade is a modle that accuratley detects faces
facefound = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')                                           
count = 0                                                                               # to keep track of the images captured
nameID = str(input("enter the name of the folder: ")).lower()                         
path = 'images/'+nameID                                                                     

is_exist = os.path.exists(path)                                                         # creates a file with the given name

if is_exist:                                                                            # if the file with the same name alredy exists then we will ask to choose a different name
    print("name of the file alredy taken")
    nameID = str(input("enter the file name again: "))
else:
    os.makedirs(path)
while True:     
    ret,frame = video.read()                                                            
    faces = facefound.detectMultiScale(frame,1.3, 5)                                    # helps in detecting faces
    for x,y,w,h in faces:                               
        count = count + 1
        name = './images/'+nameID+'/'+str(count)+'.jpg'                                 # captures and store the images in jpg format
        print("Creating Images..." + name)
        cv2.imwrite(name, frame[y:y+h, x:x+w])                                          # to create a rectangle around the face and only the stuff inside the rectangle is captured
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 3)
    cv2.imshow("WindowFrame", frame)
    k = cv2.waitKey(1)
    if count>700:                                                                       # if 700 images are clicked then the while loop breaks 
        break
video.release()
cv2.destroyAllWindows()                                                                 # closes the window