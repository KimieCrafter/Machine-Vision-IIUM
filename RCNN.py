import torch, torchvision
import cv2
from PIL import Image

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# Prepare the model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

preprocess = weights.transforms()
coco_labels = weights.meta["categories"] # Coco labels Dataset

model.eval() # Inference mode

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

print(f"Using device: {device}")

# Webcam Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert OpenCV BGR image to PIL RGB image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image and convert to batch tensor
    input_img = preprocess(pil_image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad(): # No training, only inference
        outputs = model(input_img) # Forward pass
        
    outputs = outputs[0] # Get the first image's output
    
    bboxes = outputs['boxes'].cpu().numpy().astype(int) # Bounding boxes
    labels = outputs['labels'].cpu().numpy()            # Class labels
    scores = outputs['scores'].cpu().numpy()            # Confidence scores
    
    # Draw all detections on the frame
    for bbox, label, score in zip(bboxes, labels, scores): 
        x1, y1, x2, y2 = bbox
        class_name = coco_labels[label]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show the frame ONCE per loop iteration
    cv2.imshow("RCNN Object Detection", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()