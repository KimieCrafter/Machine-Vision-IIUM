import torch
from torchvision import models, transforms
import cv2
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
label_list = ["peace-minilove-", "mini_love", "peace"]
num_classes = 4

# Load model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(r"Transfer_Learning/Model/Gesture/model_epoch_10.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
threshold = 0.8

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize if needed
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Preprocess
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor = transform(image_pil).unsqueeze(0).to(device) #type: ignore

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i].astype(int)
            label = label_list[labels[i]-1]  # ✅ subtract 1
            score = scores[i]

            text = f"{label}: {score:.2f}"
            cv2.putText(frame_resized, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.rectangle(frame_resized, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    cv2.imshow("Gesture Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
