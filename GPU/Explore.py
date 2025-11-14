import cv2
import numpy as np
import time

# Read image
img = cv2.imread('Img/test4.jpg')
if img is None:
    # Create test image if file doesn't exist
    img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

print(f"Original shape: {img.shape}")

# === CPU Version (for comparison) ===
start = time.time()
gray_cpu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_cpu = cv2.GaussianBlur(gray_cpu, (5, 5), 0)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time*1000:.2f}ms")

# === GPU Version ===
# Upload to GPU
gpu_img = cv2.cuda_GpuMat() # type: ignore
gpu_img.upload(img)

start = time.time()

# Color conversion on GPU
gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

# Create and apply Gaussian filter
gaussian = cv2.cuda.createGaussianFilter(
    cv2.CV_8UC1,  # Source: 8-bit single channel
    cv2.CV_8UC1,  # Destination: 8-bit single channel
    (5, 5),       # Kernel size
    0             # Sigma
)
gpu_blurred = gaussian.apply(gpu_gray)

# Download result
result = gpu_blurred.download()

gpu_time = time.time() - start
print(f"GPU time: {gpu_time*1000:.2f}ms")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
print(f"Result shape: {result.shape}")

# Display
cv2.imshow('CPU Result', blurred_cpu)
cv2.imshow('GPU Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()