import cv2
import numpy as np

image = cv2.imread('Quiz 1/img.jpg')

if image is None:
    raise ValueError("Could not read the image. Please check if 'img.jpg' exists.")


eye_crop = image[100:200, 150:350]

# Convert to RGB for display
eye_crop_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)

cv2.imshow('Original Image', image)
cv2.imshow('Cropped Eye', eye_crop)
cv2.waitKey(0)