from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition
import pyttsx3

# Load face images and encodings
jobs_image = face_recognition.load_image_file(r"C:\Users\DHARSHAN BALAJI\Downloads\objdetectioproject\IOT project\photos\jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file(r"C:\Users\DHARSHAN BALAJI\Downloads\objdetectioproject\IOT project\photos\tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

sadmona_image = face_recognition.load_image_file(r"C:\Users\DHARSHAN BALAJI\Downloads\objdetectioproject\IOT project\photos\monalisa.jpg")
sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

tesla_image = face_recognition.load_image_file(r"C:\Users\DHARSHAN BALAJI\Downloads\objdetectioproject\IOT project\photos\tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

known_face_encoding = [
    jobs_encoding,
    ratan_tata_encoding,
    sadmona_encoding,
    tesla_encoding
]

known_faces_names = [
    "jobs",
    "ratan tata",
    "sadmona",
    "tesla"
]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO(r"C:\Users\DHARSHAN BALAJI\Downloads\objdetectioproject\webcam detection\Yolo-weights\yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
]

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set to store detected classes
detected_classes = set()



import textwrap
import google.generativeai as genai
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Configure the API key
genai.configure(api_key="paste your key")

def get_reply(prompt):
    # Choose a model
    model = genai.GenerativeModel('gemini-pro')

    # Generate content based on the prompt
    response = model.generate_content(prompt)

    # Convert the response to Markdown and return it
    return response.text




while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Object detection
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Class name
            cls = int(box.cls[0])

            # Check if the detected object is a person
            if classNames[cls] == "person":
                # Crop the face from the image
                face_img = img[y1:y2, x1:x2]
                face_encoding = face_recognition.face_encodings(face_img)

                # Compare the face encoding with known faces
                if len(face_encoding) > 0:
                    match = face_recognition.compare_faces(known_face_encoding, face_encoding[0])
                    if any(match):
                        best_match_index = np.argmax(match)
                        name = known_faces_names[best_match_index]
                        cv2.putText(img, f"Hi {name}, How can I help you?", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        detected_classes.add(name)
                        # Speak the greeting
                        engine.say(f"Hi {name}, How can I help you!")
                        engine.runAndWait()

                        # Ask for user input
                        user_input = input("Need help? (Yes/No): ")
                        if user_input.lower() == 'yes':
                            prompt = input("Enter your prompt: ")
                            engine.say(get_reply(prompt))
                            engine.runAndWait()
                        


                        print(get_reply(prompt))

    # Display the webcam feed
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Write detected classes to a text file
with open('detected_classes.txt', 'w') as f:
    for item in detected_classes:
        f.write("%s\n" % item)

# Release resources
cap.release()
cv2.destroyAllWindows()
