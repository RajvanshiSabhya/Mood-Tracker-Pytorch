import cv2
from PIL import Image
import torch
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (same as val_transform in training)
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Preprocess image and detect face
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Robust detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

    if len(faces) == 0:
        return None, "No face detected. Make sure your face is clearly visible."

    # Take the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    face = gray_image[y:y+h, x:x+w]
    face_pil = Image.fromarray(face)
    tensor_image = val_transform(face_pil).unsqueeze(0).to(device)
    
    return tensor_image, None
