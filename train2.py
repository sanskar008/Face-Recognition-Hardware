import cv2
import os
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_dir = "faces"
current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# Save label-name mapping
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# Train the recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

print("Training complete!")
