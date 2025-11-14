import cv2

# Load the pre-trained Haar Cascade Classifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore


# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (Haar needs grayscale input)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using detectMultiScale()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # How much the image size is reduced at each image scale
        minNeighbors=5,     # How many neighbors each rectangle should have to be retained
        minSize=(30, 30)    # Minimum size of face
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Face: {len(faces)}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(len(faces))  # Print number of detected faces

    # Display the result
    cv2.imshow('Haar Cascade Face Detection', frame)

    # Exit when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
