

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load both trained models
RESNET_MODEL_PATH = "pneumonia_resnet_classifier.h5"
ORIGINAL_MODEL_PATH = "pneumonia_classifier.h5"

resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
original_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

def predict_pneumonia(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Updated size for ResNet
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    prediction = model.predict(img_array)
    return "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        results = {
            "resnet": predict_pneumonia(resnet_model, filepath),
            "original": predict_pneumonia(original_model, filepath)
        }
        
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
