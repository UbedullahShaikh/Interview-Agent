import cv2
import math
import mediapipe as mp


BODY_MOVEMENT_THRESHOLD = 0.015

# Constants for thresholds
EYE_CENTER_LEFT_THRESHOLD = 0.4
EYE_CENTER_RIGHT_THRESHOLD = 0.6
EYE_UP_THRESHOLD = 0.018
EYE_DOWN_THRESHOLD = 0.030
BLINK_THRESHOLD = 0.014

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        
    def process_frame(self, frame_rgb, frame_width, frame_height):
        results = self.face_mesh.process(frame_rgb)
        data = {
            'eye_direction': 'unknown',
            'blink_detected': False,
            'head_yaw': 0,
            'head_pitch': 0,
            'head_roll': 0,
            'face_landmarks': None
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            data['eye_direction'] = self._get_eye_direction(landmarks, frame_width)
            data['blink_detected'] = self._calculate_blink(landmarks)
            data['head_yaw'], data['head_pitch'], data['head_roll'] = self._calculate_head_movement(
                landmarks, frame_width, frame_height)
            data['face_landmarks'] = landmarks
            
        return data
    
    def _get_eye_direction(self, landmarks, frame_width):
        left_x = landmarks[33].x
        right_x = landmarks[263].x
        eye_center_x = (left_x + right_x) / 2

        left_height = abs(landmarks[159].y - landmarks[145].y)
        right_height = abs(landmarks[386].y - landmarks[374].y)
        avg_height = (left_height + right_height) / 2

        if eye_center_x < EYE_CENTER_LEFT_THRESHOLD:
            return 'left'
        elif eye_center_x > EYE_CENTER_RIGHT_THRESHOLD:
            return 'right'
        else:
            if avg_height < EYE_UP_THRESHOLD:
                return 'up'
            elif avg_height > EYE_DOWN_THRESHOLD:
                return 'down'
            else:
                return 'center'
        
    def _calculate_blink(self, landmarks):
        left_distance = abs(landmarks[159].y - landmarks[145].y)
        right_distance = abs(landmarks[386].y - landmarks[374].y)
        return (left_distance + right_distance) / 2 < BLINK_THRESHOLD
        
    def _calculate_head_movement(self, landmarks, frame_width, frame_height):
        def get_point(index):
            p = landmarks[index]
            return int(p.x * frame_width), int(p.y * frame_height)

        nose_x, nose_y = get_point(1)
        chin_x, chin_y = get_point(152)
        left_eye_x, left_eye_y = get_point(33)
        right_eye_x, right_eye_y = get_point(263)

        yaw = math.degrees(math.atan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x))
        pitch = math.degrees(math.atan2(chin_y - nose_y, chin_x - nose_x))
        roll = math.degrees(math.atan2(nose_y - chin_y, right_eye_x - left_eye_x))

        return yaw, pitch, roll