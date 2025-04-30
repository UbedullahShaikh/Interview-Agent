import math
import mediapipe as mp

BODY_MOVEMENT_THRESHOLD = 0.015

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.previous_body_center = None
        self.body_center_history = []
        
    def process_frame(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        data = {
            'body_movement': "No Movement",
            'body_learning_data': None
        }
        
        if results.pose_landmarks:
            data['body_movement'], data['body_learning_data'] = self._detect_body_movement(
                results.pose_landmarks.landmark)
                
        return data
    
    def _detect_body_movement(self, landmarks):
        if not landmarks:
            return "No Movement", None

        center_x = sum([landmarks[i.value].x for i in [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                                       self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                                       self.mp_pose.PoseLandmark.LEFT_HIP, 
                                                       self.mp_pose.PoseLandmark.RIGHT_HIP]]) / 4
        center_y = sum([landmarks[i.value].y for i in [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                                       self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                                       self.mp_pose.PoseLandmark.LEFT_HIP, 
                                                       self.mp_pose.PoseLandmark.RIGHT_HIP]]) / 4
        body_center = (center_x, center_y)

        if self.previous_body_center is None:
            self.previous_body_center = body_center
            return "No Movement", None

        dx = body_center[0] - self.previous_body_center[0]
        dy = body_center[1] - self.previous_body_center[1]
        distance_moved = math.hypot(dx, dy)

        self.body_center_history.append(body_center)

        if distance_moved > BODY_MOVEMENT_THRESHOLD:
            movement = "right" if dx > 0 else "left" if abs(dx) > abs(dy) else "down" if dy > 0 else "up"
            self.previous_body_center = body_center
            return movement, self.body_center_history[-5:]
        return "No Movement", None
