import cv2
import torch; 

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')

# Check version
print(f"OpenCV version: {cv2.__version__}")
print(f"Build info:\n{cv2.getBuildInformation()}")

# Check CUDA support
print(f"CUDA enabled devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

# If CUDA is working, you should see 1 or more devices
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print(f"CUDA device: {cv2.cuda.getDevice()}")
    print("✅ OpenCV with CUDA is working!")
else:
    print("❌ CUDA not detected")




