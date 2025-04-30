# For loading
import sys
import cv2

# Import MediaPipe for real-time face, hand, and pose detection
import mediapipe as mp

# To load previously saved model
from tensorflow.keras.models import load_model   

# 📊 Import the required library
import matplotlib.pyplot as plt




# # 💾 Load the Trained Model
model = load_model('face_mask_detector_model.keras')  # works perfectly


# # 🕵️‍♂️ Face Mask Detection Function
def detect_face_mask(img):
    # Reshape and normalize the input image
    img = img.reshape(1, 224, 224, 3) / 255.0
    
    # Predict using the trained model
    y_pred = model.predict(img)
    
    # Convert the prediction to binary label (0 or 1)
    label = int(y_pred[0][0] > 0.5)
    
    return label


# # 🖍️ Draw Label on Image
def draw_label(img, text, pos, bg_color):
    # Get the size of the text to be displayed
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)

    # Calculate coordinates for the background rectangle
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2

    # Draw the filled rectangle as background
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)

    # Put the text label on top of the rectangle
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)


# # 👤 Face Detection Using Mediapipe and Haar Cascades

# Load Haar cascade classifiers for detecting frontal and profile faces
frontal_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Define a function to detect faces using either Mediapipe or Haar cascades
def detect_faces_multi(img, use_mediapipe=False):
    faces = []

    # Try using Mediapipe if enabled and available
    if use_mediapipe:
        try:
            mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_face.process(img_rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
                return faces
        except ImportError:
            print("Mediapipe not installed, skipping...")

    # Fallback to Haar frontal face detection
    faces = frontal_haar.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        return faces

    # If no frontal face is found, use profile face detection
    faces = profile_haar.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    return faces

cap = cv2.VideoCapture(0)   # Access Webcam Feed
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size
    img = cv2.resize(frame, (224, 224))

    # Predict mask presence
    y_pred = detect_face_mask(img)

    # Convert frame to grayscale for Haar detection (not used here but kept for compatibility)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Mediapipe (falls back to Haar if needed)
    coods = detect_faces_multi(frame, use_mediapipe=True)

    # Draw bounding boxes around detected faces
    for x, y, w, h in coods:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw label based on prediction
    if y_pred == 0:
        draw_label(frame, "Mask", (50, 50), (0, 255, 0))     # Green for Mask
    else:
        draw_label(frame, "No Mask", (50, 50), (0, 0, 255))  # Red for No Mask

    # Display the frame
    cv2.imshow("window", frame)

    # Exit loop when 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources
cap.release()

cv2.destroyAllWindows()

