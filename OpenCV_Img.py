import cv2
import numpy as np

image = cv2.imread('Img/test4.jpg')

if image is None:
    raise ValueError("Could not read the image. Please check if 'Img/test4.jpg' exists.")

print(image.shape) # Print original image shape

# Resize Image to specific dimensions #
#######################################

resized_downwidth = 640
Resize_height = 480
Resize_points = (resized_downwidth, Resize_height)
resized_Image = cv2.resize(image, Resize_points, interpolation= cv2.INTER_LINEAR)

# Image Annotation #
####################

resized_downwidth = 640
Resize_height = 480
Resize_points = (resized_downwidth, Resize_height)
resized_Image_Annote = cv2.resize(image, Resize_points, interpolation= cv2.INTER_LINEAR)

# Draw Rectangle

start_point =(225,105)  # x1,y1 (Top-left corner)
end_point =(415,325)    # x2,y2 (Bottom-right corner)

cv2.rectangle(resized_Image_Annote, start_point, end_point, (0, 0, 255), thickness= 3, lineType=cv2.LINE_8)

# Draw Line

cv2.line(resized_Image_Annote, (100,105), (100,325), (255,0,0), 5) # (X1,Y1), (X2,Y2), (B,G,R), Thickness

# Draw Circle

cv2.circle(resized_Image_Annote, center=(320,240), radius= 100, color=(0,255,0), thickness= 3) # Center, Radius, Color, Thickness

# Annotate Text

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (255,255,0)     # Turquoise Color
pos = (250,50)               # Position (X,Y)
text = 'Edryana'             # Text

cv2.putText(resized_Image_Annote, text=text, org=pos, fontFace=font, fontScale=font_scale, color=font_color, thickness=font_thickness, lineType=cv2.LINE_AA)

# image Filtering #
####################

# Downsize the Image & Apply Kernal Blur

resized_downwidth_Blur = 640
Resize_height_Blur = 480
Resize_points_Blur = (resized_downwidth_Blur, Resize_height_Blur)
resized_Image_Blur = cv2.resize(image, Resize_points_Blur, interpolation= cv2.INTER_LINEAR)

kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

Identity = cv2.filter2D(src=resized_Image_Blur, ddepth=-1, kernel=kernel1)

# Other Blur Techniques

blur = cv2.blur(resized_Image,(5,5))
# blur = cv2.GaussianBlur(resized_Image,(5,5),0)
# blur = cv2.medianBlur(resized_Image,5)
# blur = cv2.bilateralFilter(resized_Image,9,75,75)

# Cropping the image

cropped_image = resized_Image[105:325, 225:415] # [y1:y2, x1:x2]

# Rotate the Image Resized

Resize_height, Resize_width = resized_Image.shape[:2]
center = (Resize_width/2, Resize_height/2)

rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
rotated_image = cv2.warpAffine(src=resized_Image, M=rotate_matrix, dsize=(Resize_width, Resize_height))

# Edge Detection using Sobel() function

img_gray = cv2.cvtColor(Identity, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Edge Detection using Canny() function

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # 1 = Lower Threshold, 2 = Upper Threshold

# Image Morphological Transformations  #
########################################

kernel2 = np.ones((5,5),np.uint8) 

# Erosion (Make Pixel Thinner)

erosion = cv2.erode(resized_Image,kernel2,iterations = 1)

# Dillation (Make Pixel Thicker)

dilation = cv2.dilate(resized_Image,kernel2,iterations = 1)

# Opening (Erosion followed by Dilation)

opening = cv2.morphologyEx(resized_Image, cv2.MORPH_OPEN, kernel2) # To remove BG Noise

# Closing (Dilation followed by Erosion)

closing = cv2.morphologyEx(resized_Image, cv2.MORPH_CLOSE, kernel2) # To remove FG Noise

# Show all the images  #
########################

print(resized_Image.shape) # Print new image shape

cv2.imshow('Edryana Resized', resized_Image)
cv2.imshow('Edryana Annotation Box', resized_Image_Annote)
cv2.imshow('Edryana Face', cropped_image)
cv2.imshow('Edryana Rotated', rotated_image)
cv2.imshow('Edryana Identity', Identity)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.imshow('Edryana Canny', edges)
cv2.imshow('Edryana Gaussian Blur', blur)
cv2.imshow('Edryana Erosion', erosion)
cv2.imshow('Edryana Dilation', dilation)
cv2.imshow('Edryana Opening', opening)
cv2.imshow('Edryana Closing', closing)
cv2.waitKey()

cv2.destroyAllWindows()