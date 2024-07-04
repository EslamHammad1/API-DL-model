from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your .h5 model file
model = load_model('model.h5')

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def home():
    return "<h1>WELCOME</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file was properly uploaded to our endpoint
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    image = request.files['image']

    # Save the file to disk for debugging purposes
    image.save('uploaded_image.jpg')

    # Preprocess the image
    img_array = preprocess_image('uploaded_image.jpg')

    # Predict using your loaded model
    prediction = model.predict(img_array)

    # Return prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
