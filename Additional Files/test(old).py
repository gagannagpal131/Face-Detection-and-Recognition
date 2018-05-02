import numpy as np
import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
im = []
for i in range(10):
    ret,im = cap.read() # return a single frame in variable `frame`
    cv2.imwrite('opencv'+str(i)+'.png', im)

# Convert the captured frame into grayscale
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# Get all face from the video frame
faces = faceCascade.detectMultiScale(gray, 2,5)

# For each face in faces
i=0
print(faces)
print("hello")
for(x,y,w,h) in faces:
    print("hello2")
    # Create rectangle around the face
    cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

    # Recognize the face belongs to which ID
    Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    i=i+1
    print("FACE %s" % (i))
    print("ID %s" % Id)
    print("Confidence %s" % confidence)
    print()
    # Check the ID if exist
    if(Id == 1):
        name = "Gagandeep Singh Nagpal"
    #If not exist, then it is Unknown
    elif(Id == 2):
        name = "Jappreet Singh"
    elif(Id == 3):
        name = "Nishant Rao"
    elif(Id == 5):
        name = "MY LOVE"    
    else:
        print(Id)
        name = "Unknown"

    print("Final ID selected: %s "  %name)
    print("----")
    # Put text describe who is in the picture cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)

# Display the video frame with the bounded rectanglecv2.imshow('im',im)
