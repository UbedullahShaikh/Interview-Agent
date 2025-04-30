from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='data/models/yolov8n.pt'):
        self.model = YOLO(model_path)
        
    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls)]
                confidence = float(box.conf[0])
                
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'box': [x1, y1, x2, y2]
                })
                
        return detections