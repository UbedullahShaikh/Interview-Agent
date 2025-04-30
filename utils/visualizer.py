import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'text': (0, 255, 0),  # Green
            'warning': (0, 0, 255),  # Red
            'highlight': (255, 0, 0)  # Blue
        }
    
    def draw_all(self, frame, face_data, pose_data, objects, emotion):
        """Draw all visual elements on the frame"""
        self._draw_face_info(frame, face_data)
        self._draw_pose_info(frame, pose_data)
        self._draw_objects(frame, objects)
        self._draw_emotion(frame, emotion, face_data.get('face_landmarks', []), frame.shape[1], frame.shape[0])
        self._draw_eye_landmarks(frame, face_data.get('eye_landmarks', []))
    
    def _draw_face_info(self, frame, face_data):
        """Draw face-related information"""
        info = [
            f'Eye: {face_data["eye_direction"]}',
            f'Yaw: {face_data["head_yaw"]:.1f}',
            f'Pitch: {face_data["head_pitch"]:.1f}',
            f'Roll: {face_data["head_roll"]:.1f}',
            'Blink: YES' if face_data["blink_detected"] else 'Blink: NO'
        ]
        
        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
    
    def _draw_pose_info(self, frame, pose_data):
        """Draw body pose information"""
        cv2.putText(frame, f'Body: {pose_data["body_movement"]}', 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
    
    def _draw_objects(self, frame, objects):
        """Draw detected objects"""
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{obj['label']} {obj['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    def _draw_emotion(self, frame, emotion, landmarks, frame_width, frame_height):
        """Draw emotion information"""
        if landmarks:
            x_min = int(min([lm.x for lm in landmarks]) * frame_width)
            y_min = int(min([lm.y for lm in landmarks]) * frame_height)
            cv2.putText(frame, f'Emotion: {emotion}', 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_eye_landmarks(self, frame, eye_landmarks):
        """Draw eye landmarks"""
        for eye in eye_landmarks:
            for point in eye:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green circle