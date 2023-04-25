import face_recognition
import cv2
import numpy as np
import os
import csv
import datetime

encodings_file = 'encodings.txt'
encodings = []
with open(encodings_file, 'r') as f:
    for line in f:
        encoding = np.array(line.split(','), dtype=np.float64)
        encodings.append(encoding)

image_folder = './images'

image_files = [f for f in os.listdir(
    image_folder) if os.path.isfile(os.path.join(image_folder, f))]

csv_file = 'Attendance_log.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Detection Time'])

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(encodings, face_encoding)

        match_index = None
        for j, match in enumerate(matches):
            if match:
                match_index = j
                break

        if match_index is not None:
            photo_name = image_files[match_index]
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            photo_name_list = photo_name.split('.')
            print(f"Match found: {photo_name_list[0]} at {current_time}")

            top, right, bottom, left = face_locations[i]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, photo_name_list[0], (left, top - 10),
                        font, 0.5, (0, 0, 255), 2)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([photo_name_list[0], current_time])
        else:
            print("No match found")

    cv2.imshow('Face Recognition-based Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
