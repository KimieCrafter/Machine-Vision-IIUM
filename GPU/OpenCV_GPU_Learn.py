import cv2


print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

image = cv2.imread('Img/test4.jpg')

gpu_img = cv2.cuda_GpuMat() #type:ignore
gpu_img.upload(image)

resized_downwidth = 640
Resize_height = 480
Resize_points = (resized_downwidth, Resize_height)
resized_Image = cv2.cuda.resize(gpu_img, Resize_points, interpolation= cv2.INTER_LINEAR)
 
 
cv2.imshow('Resized Image on GPU', resized_Image.download())
cv2.waitKey(0)