import cv2
import numpy as np
import time
import dlib
from datetime import datetime
import os
from scipy.spatial import distance

class EmotionDetector:
    def __init__(self):
        # Further adjusted thresholds
        self.smile_threshold = 0.15  # Even more sensitive to smiles
        self.eye_ar_threshold = 0.15  # More sensitive to eye changes
        self.mouth_ar_threshold = 0.30  # More sensitive to mouth movements
        self.eyebrow_threshold = 0.15  # More sensitive to eyebrow positions
        self.smile_intensity_threshold = 0.06  # More sensitive to smile intensity
        self.furrow_threshold = 0.20  # More sensitive to brow furrows
        
    def eye_aspect_ratio(self, eye_points):
        # Compute the euclidean distances between the vertical eye landmarks
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        # Compute the euclidean distance between the horizontal eye landmarks
        C = distance.euclidean(eye_points[0], eye_points[3])
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth_points):
        # Compute the euclidean distances between the vertical mouth landmarks
        A = distance.euclidean(mouth_points[2], mouth_points[10])
        B = distance.euclidean(mouth_points[4], mouth_points[8])
        # Compute the euclidean distance between the horizontal mouth landmarks
        C = distance.euclidean(mouth_points[0], mouth_points[6])
        # Mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar
    
    def eyebrow_position(self, eyebrow_points, eye_points):
        # Calculate the average height of eyebrow relative to eye
        eyebrow_y = np.mean([p[1] for p in eyebrow_points])
        eye_y = np.mean([p[1] for p in eye_points])
        return eye_y - eyebrow_y
    
    def detect_furrowed_brow(self, inner_brows, outer_brows):
        # More precise furrow detection
        left_angle = np.arctan2(inner_brows[0][1] - outer_brows[0][1],
                               inner_brows[0][0] - outer_brows[0][0])
        right_angle = np.arctan2(inner_brows[1][1] - outer_brows[1][1],
                                inner_brows[1][0] - outer_brows[1][0])
        # Check if angles indicate significant furrowing and are similar (symmetric)
        angle_diff = abs(left_angle - right_angle)
        return (abs(left_angle) > self.furrow_threshold or 
                abs(right_angle) > self.furrow_threshold) and angle_diff < 0.2
    
    def measure_smile_intensity(self, mouth_points):
        # Calculate smile intensity based on curve and width
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        middle_top = mouth_points[3]
        
        # Calculate smile curve
        curve = (left_corner[1] + right_corner[1])/2 - middle_top[1]
        # Calculate smile width
        width = distance.euclidean(left_corner, right_corner)
        
        return curve / width
    
    def detect_genuine_smile(self, mouth_points):
        # Enhanced smile detection with intensity
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        middle_top = mouth_points[3]
        middle_bottom = mouth_points[9]
        
        # Check if corners are lifted
        corner_height = (left_corner[1] + right_corner[1]) / 2
        middle_height = (middle_top[1] + middle_bottom[1]) / 2
        
        # Calculate symmetry with more tolerance
        left_side = abs(left_corner[1] - middle_top[1])
        right_side = abs(right_corner[1] - middle_top[1])
        symmetry = abs(left_side - right_side) / max(left_side, right_side)
        
        intensity = self.measure_smile_intensity(mouth_points)
        is_symmetric = symmetry < 0.4  # More tolerant symmetry check
        
        return corner_height < middle_height and is_symmetric, intensity
    
    def detect_emotion(self, landmarks, smile_detected):
        # Extract facial points...
        left_eye = [(landmarks.part(36+i).x, landmarks.part(36+i).y) for i in range(6)]
        right_eye = [(landmarks.part(42+i).x, landmarks.part(42+i).y) for i in range(6)]
        mouth = [(landmarks.part(48+i).x, landmarks.part(48+i).y) for i in range(12)]
        
        left_eyebrow = [(landmarks.part(17+i).x, landmarks.part(17+i).y) for i in range(5)]
        right_eyebrow = [(landmarks.part(22+i).x, landmarks.part(22+i).y) for i in range(5)]
        
        inner_brows = [(landmarks.part(21).x, landmarks.part(21).y),
                      (landmarks.part(22).x, landmarks.part(22).y)]
        outer_brows = [(landmarks.part(17).x, landmarks.part(17).y),
                      (landmarks.part(26).x, landmarks.part(26).y)]
        
        # Calculate metrics
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = self.mouth_aspect_ratio(mouth)
        
        is_genuine_smile, smile_intensity = self.detect_genuine_smile(mouth)
        
        left_eyebrow_pos = self.eyebrow_position(left_eyebrow, left_eye)
        right_eyebrow_pos = self.eyebrow_position(right_eyebrow, right_eye)
        avg_eyebrow_pos = (left_eyebrow_pos + right_eyebrow_pos) / 2
        
        is_furrowed = self.detect_furrowed_brow(inner_brows, outer_brows)
        
        # Revised emotion classification with better balance
        
        # First check strong negative emotions
        if is_furrowed:
            if mar > self.mouth_ar_threshold * 1.2:
                return "Frustrated", (0, 0, 255), mouth
            if avg_eyebrow_pos < self.eyebrow_threshold * 0.8:
                return "Angry", (0, 0, 255), mouth
            return "Concerned", (255, 165, 0), mouth
            
        # Check for surprise (wide eyes and mouth)
        if mar > self.mouth_ar_threshold * 1.2 and avg_ear > self.eye_ar_threshold * 1.2:
            return "Surprised", (255, 255, 0), mouth
            
        # Check for sadness and disappointment
        if avg_eyebrow_pos < self.eyebrow_threshold * 0.85:
            if mar < self.mouth_ar_threshold * 0.8:
                return "Disappointed", (255, 0, 255), mouth
            return "Sad", (0, 0, 255), mouth
            
        # Check for confusion
        if mar > self.mouth_ar_threshold and avg_ear < self.eye_ar_threshold * 0.9:
            if avg_eyebrow_pos < self.eyebrow_threshold:
                return "Confused", (255, 128, 0), mouth
                
        # Check for focused/concentrated expression
        if avg_ear < self.eye_ar_threshold * 0.9 and not smile_detected:
            if avg_eyebrow_pos < self.eyebrow_threshold:
                return "Focused", (128, 128, 128), mouth
                
        # Now check for positive emotions
        if smile_detected or smile_intensity > self.smile_intensity_threshold:
            if smile_intensity > self.smile_intensity_threshold * 1.5:
                if avg_ear < self.eye_ar_threshold * 0.9:  # Squinted eyes
                    return "Joyful", (0, 255, 0), mouth
                return "Very Happy", (0, 255, 0), mouth
            elif avg_ear < self.eye_ar_threshold:
                return "Amused", (0, 255, 128), mouth
            elif smile_intensity > self.smile_intensity_threshold:
                return "Content", (0, 255, 128), mouth
        
        # Check for neutral expression
        if (abs(mar - self.mouth_ar_threshold) < 0.05 and 
            abs(avg_ear - self.eye_ar_threshold) < 0.05 and 
            abs(avg_eyebrow_pos - self.eyebrow_threshold) < 0.05):
            return "Neutral", (255, 255, 255), mouth
            
        # Slight variations from neutral
        if avg_ear < self.eye_ar_threshold:
            return "Pensive", (128, 128, 255), mouth
        if mar > self.mouth_ar_threshold:
            return "Interested", (128, 255, 128), mouth
            
        return "Neutral", (255, 255, 255), mouth

class FaceDetector:
    def __init__(self):
        # Create output directory for recordings and screenshots
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Initialize facial landmark predictor and emotion detector
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.emotion_detector = EmotionDetector()
        
        # Initialize parameters
        self.mode = 'emotion'  # Default mode is now emotion detection
        self.recording = False
        self.out = None
        self.start_time = time.time()
        self.frame_count = 0
        self.show_landmarks = True  # Always show landmarks for emotion detection
        
    def calculate_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return fps
    
    def draw_landmarks(self, frame, face):
        landmarks = self.predictor(frame, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        return landmarks
    
    def start_recording(self, frame):
        if not self.recording:
            filename = os.path.join(self.output_dir, 
                                  f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
            self.recording = True
    
    def stop_recording(self):
        if self.recording:
            self.out.release()
            self.recording = False
    
    def take_screenshot(self, frame):
        filename = os.path.join(self.output_dir, 
                              f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        cv2.imwrite(filename, frame)
    
    def visualize_smile_points(self, frame, mouth_points, color):
        # Visualize key smile detection points
        for point in mouth_points:
            cv2.circle(frame, (point[0], point[1]), 2, color, -1)
        
        # Draw smile curve
        mouth_hull = cv2.convexHull(np.array(mouth_points))
        cv2.drawContours(frame, [mouth_hull], -1, color, 1)
        
        # Highlight corners and middle points
        cv2.circle(frame, mouth_points[0], 3, (255, 255, 0), -1)  # Left corner
        cv2.circle(frame, mouth_points[6], 3, (255, 255, 0), -1)  # Right corner
        cv2.circle(frame, mouth_points[3], 3, (0, 255, 255), -1)  # Top middle
        cv2.circle(frame, mouth_points[9], 3, (0, 255, 255), -1)  # Bottom middle

    def detect_faces(self):
        cap = cv2.VideoCapture(0)
        
        # Create window and trackbars
        cv2.namedWindow('Face Detection')
        cv2.createTrackbar('Min Neighbors', 'Face Detection', 5, 20, lambda x: None)
        cv2.createTrackbar('Scale Factor (%)', 'Face Detection', 110, 200, lambda x: None)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get trackbar values
            min_neighbors = cv2.getTrackbarPos('Min Neighbors', 'Face Detection')
            scale_factor = cv2.getTrackbarPos('Scale Factor (%)', 'Face Detection') / 100
            if scale_factor < 1.1:
                scale_factor = 1.1
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            
            for (x, y, w, h) in faces:
                # Get face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect smile in the face region
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=25,
                    minSize=(int(w*0.3), int(h*0.1))
                )
                smile_detected = len(smiles) > 0
                
                # Convert face coordinates to dlib rectangle
                face_rect = dlib.rectangle(x, y, x+w, y+h)
                
                # Draw landmarks and get them for emotion detection
                landmarks = self.draw_landmarks(frame, face_rect)
                
                # Detect emotion
                emotion, color, mouth_points = self.emotion_detector.detect_emotion(landmarks, smile_detected)
                
                # Draw face rectangle with emotion color
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion text
                cv2.putText(frame, emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Visualize smile detection points
                self.visualize_smile_points(frame, mouth_points, color)
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display recording status
            if self.recording:
                cv2.putText(frame, 'REC', (frame.shape[1]-100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Record frame if recording is active
            if self.recording:
                self.out.write(frame)
            
            # Display the frame
            cv2.imshow('Face Detection', frame)
            
            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Toggle recording
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(frame)
            elif key == ord('s'):  # Take screenshot
                self.take_screenshot(frame)
        
        # Release resources
        if self.recording:
            self.stop_recording()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.detect_faces() 