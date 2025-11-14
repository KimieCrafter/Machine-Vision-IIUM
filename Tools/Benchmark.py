import time
import torch
import cv2
import numpy as np

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# ----- Setup -----
device = torch.device("cuda")
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device)
model.eval()

# Sample input
img = cv2.imread("Img/test4.jpg")

# Optional: warm-up GPU
for _ in range(5):
    dummy = torch.randn(1, 3, 640, 640).to(device)
    _ = model(dummy)

# ----- Benchmark -----
num_runs = 50
times = []

for _ in range(num_runs):
    start = time.time()

    # 1️⃣ Preprocessing (OpenCV)
    resized = cv2.resize(img, (640, 640)) # type: ignore
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # 2️⃣ Inference (PyTorch GPU)
    with torch.no_grad():
        outputs = model(tensor)

    # 3️⃣ Postprocessing (OpenCV)
    # Example: just draw random boxes to simulate work
    for i in range(5):
        cv2.rectangle(resized, (10*i, 10*i), (50*i, 50*i), (0, 255, 0), 2)

    end = time.time()
    times.append(end - start)

avg_time = sum(times) / num_runs
print(f"Average total time per frame: {avg_time:.4f} s")
print(f"→ Approx FPS: {1/avg_time:.2f}")