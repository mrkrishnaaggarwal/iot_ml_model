from flask import Flask, jsonify, request
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models
from PIL import Image
import cv2
import math
import base64

app = Flask(__name__)

# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights=True)
model.eval()

# Function to load the classification model
def load_model(model_path):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)  # Assuming 7 classes for classification
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load the classification model
classification_model = load_model('resnet50_model.pth')  # Path to your classification model
#hi

# Function to preprocess the input image
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Function to perform prediction
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Route to detect and classify the image
@app.route('/detect_and_classify', methods=['POST'])
def detect_and_classify():
    data = request.get_json()  # Get the JSON data from the request
    base64_image = data.get('image')  # Extract the base64 image

    if not base64_image:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    image_data = base64.b64decode(base64_image)

    # Save the decoded image to a temporary file
    image_path = "tmp/input_image.jpg"
    with open(image_path, 'wb') as f:
        f.write(image_data)

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    origin = (width // 2, height)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_rgb).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    min_distance = float('inf')
    closest_box = None
    closest_score = None

    for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
        if score > 0.2:
            x1, y1, x2, y2 = box.int().numpy()
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            distance = math.sqrt((center_x - origin[0]) ** 2 + (center_y - origin[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_box = (x1, y1, x2, y2)
                closest_score = score.item()

    if closest_box is None:
        return jsonify({'error': 'No suitable object detected'}), 400

    x1, y1, x2, y2 = closest_box
    cropped_image = image_rgb[y1:y2, x1:x2]
    crop_filename = "tmp/closest_crop.jpg"
    cv2.imwrite(crop_filename, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

    image_tensor = preprocess_image(crop_filename)
    class_names = ['Aluminium', 'Carton', 'E-waste', 'Glass', 'Organic_Waste', 'Plastics', 'Wood']
    predicted_class = predict(classification_model, image_tensor, class_names)

    response = {
        'x': int((x1 + x2) // 2),  # Convert to int for JSON serialization
        'y': int((y1 + y2) // 2),  # Convert to int for JSON serialization
        'predicted_class': predicted_class,
        'confidence': float(closest_score)  # Convert to float for JSON serialization
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
