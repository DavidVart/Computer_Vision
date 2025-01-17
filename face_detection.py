import cv2
import numpy as np
import time
import dlib
from datetime import datetime
import os

class FaceDetector:
    def __init__(self):
        # Create output directory for recordings and screenshots
        self.output_dir = 'screenshots'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Initialize facial landmark predictor
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize parameters
        self.mode = 'face'  # Current detection mode
        self.recording = False
        self.out = None
        self.start_time = time.time()
        self.frame_count = 0
        self.show_landmarks = False
        
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
            if scale_factor < 1.1:  # Minimum scale factor
                scale_factor = 1.1
            
            # Always detect faces first for smile detection
            faces = self.face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            
            # Detection based on current mode
            if self.mode == 'face':
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if self.show_landmarks:
                        face_rect = dlib.rectangle(x, y, x+w, y+h)
                        self.draw_landmarks(frame, face_rect)
                        
            elif self.mode == 'eyes':
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray, scale_factor, min_neighbors)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                    
            elif self.mode == 'smile':
                for (x, y, w, h) in faces:
                    # Draw face rectangle in smile mode too
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    
                    # Define ROI for smile detection (lower half of face)
                    roi_gray = gray[y + h//2:y + h, x:x + w]
                    roi_color = frame[y + h//2:y + h, x:x + w]
                    
                    # Detect smiles with adjusted parameters
                    smiles = self.smile_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=max(min_neighbors, 25),  # Increase minimum neighbors for smile
                        minSize=(int(w*0.3), int(h*0.1))  # Minimum size relative to face
                    )
                    
                    for (sx, sy, sw, sh) in smiles:
                        # Adjust coordinates to account for ROI offset
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display mode and recording status
            cv2.putText(frame, f'Mode: {self.mode}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
            elif key == ord('m'):  # Change mode
                modes = ['face', 'eyes', 'smile']
                current_index = modes.index(self.mode)
                self.mode = modes[(current_index + 1) % len(modes)]
            elif key == ord('r'):  # Toggle recording
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(frame)
            elif key == ord('s'):  # Take screenshot
                self.take_screenshot(frame)
            elif key == ord('l'):  # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
        
        # Release resources
        if self.recording:
            self.stop_recording()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.detect_faces() 