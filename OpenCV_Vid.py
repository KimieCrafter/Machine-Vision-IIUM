import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)                       # Webcam source (0) 
# cap = cv2.VideoCapture('Vid/test1.mp4')       # Video file source

while True:
    ret, frame = cap.read()

    # Any processing on the frame can be done here #
    ################################################
    
    edges = cv2.Canny(frame,50,150)

    if not ret:
        break

    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()