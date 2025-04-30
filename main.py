import cv2
import time
import json
from detectors.face_detector import FaceDetector
from detectors.pose_detector import PoseDetector
from detectors.object_detector import ObjectDetector
from detectors.emotion_detector import EmotionDetector
from utils.visualizer import Visualizer

class InterviewAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.pose_detector = PoseDetector()
        self.object_detector = ObjectDetector()
        self.emotion_detector = EmotionDetector('data/models/fer2013_mini_XCEPTION.102-0.66.hdf5')
        self.visualizer = Visualizer()
        self.frames_data = []
        self.start_time = time.time()
        
    def run(self):
        cap = cv2.VideoCapture(0)
        print("Starting video capture... Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width, _ = frame.shape
            
            # Process all detectors
            face_data = self.face_detector.process_frame(frame_rgb, frame_width, frame_height)
            pose_data = self.pose_detector.process_frame(frame_rgb)
            objects = self.object_detector.detect_objects(frame)
            emotion = self._detect_emotion(frame, face_data.get('face_landmarks'), frame_width, frame_height)
            
            # Store frame data
            self._store_frame_data(face_data, pose_data, objects, emotion)
            
            # Visualize
            self.visualizer.draw_all(frame, face_data, pose_data, objects, emotion)
            
            cv2.imshow('Interview Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self._save_data()
        
    def _detect_emotion(self, frame, landmarks, frame_width, frame_height):
        if not landmarks:
            return "Unknown"
            
        x_min = int(min([lm.x for lm in landmarks]) * frame_width)
        y_min = int(min([lm.y for lm in landmarks]) * frame_height)
        x_max = int(max([lm.x for lm in landmarks]) * frame_width)
        y_max = int(max([lm.y for lm in landmarks]) * frame_height)
        
        face_img = frame[y_min:y_max, x_min:x_max]
        if face_img.size > 0:
            return self.emotion_detector.detect_emotion(face_img)
        return "Unknown"
        
    def _store_frame_data(self, face_data, pose_data, objects, emotion):
        self.frames_data.append({
            'timestamp': time.time() - self.start_time,
            'eye_tracking': {
                'direction': face_data['eye_direction'],
                'blink_detected': face_data['blink_detected']
            },
            'head_tracking': {
                'yaw': face_data['head_yaw'],
                'pitch': face_data['head_pitch'],
                'roll': face_data['head_roll']
            },
            'body_tracking': {
                'movement': pose_data['body_movement'],
                'position_history': pose_data['body_learning_data']
            },
            'emotion': {
                'detected_emotion': emotion
            },
            'detections': {
                'yolo': objects
            }
        })
        
    def _save_data(self):
        with open('data/outputs/interview_tracker_with_yolo_emotions.json', 'w') as f:
            json.dump(self.frames_data, f, indent=4)
        print("Data saved to interview_tracker_with_yolo_emotions.json")

if __name__ == "__main__":
    analyzer = InterviewAnalyzer()
    analyzer.run()