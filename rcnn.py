import torch
import torchvision
import cv2
from torchvision import transforms
from PIL import Image

# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# COCO Labels
COCO_INSTANCE_CATEGORY_NAMES = [
'_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Input/output video
input_video = 'input_videos/neutral1.mp4'
output_video = 'rcnn_output_video.avi'
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
      
# Convert frame to PIL, then transform
image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
img_tensor = transform(image_pil).unsqueeze(0)

# Detect
with torch.no_grad():
    predictions = model(img_tensor)[0]

# Draw detections
for i in range(len(predictions["boxes"])):
    score = predictions["scores"][i].item()
        if score < 0.5:
            continue
        box = predictions["boxes"][i].int().tolist()
        label = COCO_INSTANCE_CATEGORY_NAMES[predictions["labels"][i]]
        confidence = f"{score:.2f}"

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        text = f"{label} {confidence}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (box[0], box[1] - text_h - 4), (box[0] + text_w, box[1]), (0, 255, 0), -1)
        cv2.putText(frame, text, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

  out.write(frame)
  frame_count += 1
  print(f"Processed frame: {frame_count}", end="\r")

cap.release()
out.release()
print("\nVideo processing complete. Saved to:", output_video)
