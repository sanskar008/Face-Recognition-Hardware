import cv2
import os

# Create directory to store the images
person_name = input("Enter the name of the person: ")
person_dir = os.path.join('faces', person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Initialize the webcam (camera index 0)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize count and capture 30 images
count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop and save the face images
        face = gray[y:y+h, x:x+w]
        face_filename = os.path.join(person_dir, f'{count}.jpg')
        cv2.imwrite(face_filename, face)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Capture Face Images', frame)

    # Break if 30 images are captured
    if count >= 30:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
