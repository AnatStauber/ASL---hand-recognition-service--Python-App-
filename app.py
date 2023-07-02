from flask import Flask, request, jsonify
import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
# import argparse
import albumentations
import torch.nn.functional as F
# import time
import cnn_model
import base64
import os
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO


app = Flask(__name__)
CORS(app)

# Load label binarizer
lb = joblib.load('./data/output/lb.pkl')

# Load the model
model = cnn_model.CustomCNN().cpu()
model.load_state_dict(torch.load('./data/output/model.pth'))
print(model)
print('Model loaded')

# Define the transformation pipeline
aug = albumentations.Compose([
    albumentations.Resize(224, 224, always_apply=True),
])

def save_base64_image(base64_string, file_path):
    # Remove the "data:image/jpeg;base64," prefix from the base64 string
    encoded_data = base64_string.split(",")[1]

    # Decode the base64 string into bytes
    decoded_data = base64.b64decode(encoded_data)

    # Create a PIL Image object from the decoded bytes
    image = Image.open(BytesIO(decoded_data))

    # Save the image as a JPEG file
    image.save(file_path, "JPEG")


def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224,224))
    # print ("hand line 37")
    # print (hand)
    output_path = "./data/temp/hand.jpg"
    cv2.imwrite(output_path, hand)
    print("Image saved successfully.")
    return hand

@app.route('/')
def hello_world():
    return 'Hello, World!'
    
@app.route('/api/detect-gesture', methods=['GET'])
def test():
    return 'test works'

@app.route('/api/detect-gesture', methods=['POST'])
def detect_gesture():
   # Get the image data from the request
    image_data = request.get_json()['image']
    # print("image_data line 45 " + image_data)
    # Save the image data to a file
    image_path = './data/temp/image.jpg'
    
    save_base64_image(image_data, image_path)

    # Load the saved image
    image = cv2.imread(image_path)
    
    if image is not None:
        print("cv read:", str(image))
    else:
        print("Failed to read image")

    cv2.rectangle(image, (100, 100), (324, 324), (20, 34, 255), 2)
    hand = hand_area(image)
    image = hand

    # Apply transformations to the image
    image = aug(image=np.array(image))['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).cpu()
    image = image.unsqueeze(0)

     # Perform gesture detection
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    detected_gesture = lb.classes_[preds]

    # Delete the temporary image file
    os.remove(image_path)

    # Return the detected gesture as a JSON response
    return jsonify({'gesture': detected_gesture})

# if __name__ == '__main__':
#     app.run()
