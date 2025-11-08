from ultralytics import YOLO
model = YOLO('yolov8x')
results6 = model.predict('input_videos/neutral1.mp4',save=True)
print(results6[0])
print('=====================================')
for box in results6[0].boxes:
    print(box)
