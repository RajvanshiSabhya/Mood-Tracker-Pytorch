from flask import Flask, request, jsonify, render_template
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
from detector import preprocess_image, device

# ---------------------------
# Define CNN model (same as training)
# ---------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128*6*6,128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,num_classes)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout4(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------------------------
# Load model
# ---------------------------
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

def predict_emotion(image):
    tensor_image, error = preprocess_image(image)
    if error:
        return None, error
    
    with torch.no_grad():
        outputs = model(tensor_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return emotion_labels[predicted_class], None

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/analyze", methods=["POST"])
def analyze_mood():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"mood":"Error","recommendations":["Invalid image. Try again."]})

    mood, error = predict_emotion(image)
    if error:
        return jsonify({"mood":"Error","recommendations":[error]})

    recommendations = {
        "Happy": ["Keep smiling!", "Celebrate your happiness!"],
        "Sad": ["Take a walk outside.", "Talk to a trusted friend."],
        "Angry": ["Take a deep breath and count to 10.", "Step outside for fresh air."],
        "Neutral": ["Maintain your balance and focus.", "Enjoy your steady mood."],
        "Surprise": ["Embrace the unexpected moment!", "Share the surprise with others."],
        "Fear": ["Ground yourself by breathing deeply.", "Talk to someone you trust."],
        "Disgust": ["Refocus on things you enjoy.", "Take a moment to clear your mind."]
    }

    return jsonify({
        "mood": mood,
        "recommendations": random.sample(recommendations.get(mood, ["Stay positive!"]), k=2)
    })

if __name__ == "__main__":
    app.run(debug=True)
