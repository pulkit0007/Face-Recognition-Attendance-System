import face_recognition
import os

image_folder = './images'

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

encodings = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    encodings.append(encoding)

with open('encodings.txt', 'w') as f:
    for encoding in encodings:
        f.write(','.join(map(str, encoding)) + '\n')
