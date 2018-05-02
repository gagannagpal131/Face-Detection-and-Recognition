# Import OpenCV2 for image processing
import cv2
from datetime import datetime
f = open("codeoutput.txt","w+")
# Import numpy for matrices calculations
import numpy as np
print("1")
print(str(datetime.now()))
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load the trained mode
recognizer.read("trainer/trainer.yml")
# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);
# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)
#cam.set(cv2.CAP_PROP_FPS, 30)
# print(cam.get(cv2.CAP_PROP_FPS))
# cam.set(cv2.CAP_PROP_FPS, 10)
i=0
# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id = recognizer.predict(gray[y:y+w,x:x+h])
        print(Id)
        # Check the ID if exist
        if(Id[0] == 1 and Id[1] <= 50):
            name = "Gagandeep Singh Nagpal"
        #If not exist, then it is Unknown
        elif(Id[0] == 2 and Id[1] <= 50):
            name = "Gagandeep Singh"

        elif(Id[0] == 3 and Id[1] <= 50):
            name = "Jappreet Singh"

        elif(Id[0] == 4 and Id[1] <= 50):
            name = "Nishant Rao"

        else:
            name = "Unknown"

        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(name), (x,y-40), font, 2, (255,255,255), 3)
        f.write("ID from iteration: %s " % Id[0])
        f.write("Distance from iteration: %s " % Id[1])
        f.write("Name found is: %s " % name)
        f.write("\n------------------------------------\n")

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im)

    # If 'q' is pressed, close program
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
