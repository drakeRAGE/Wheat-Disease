from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
import os
from base64 import b64encode
import pickle

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and label binarizer
model = load_model("wheatmodel1.h5")
lb = pickle.loads(open("label", "rb").read())

# Define a function to preprocess the input image
def preprocess_image(image):
    # Resize the image to 224x224 pixels
    image = cv2.resize(image, (224, 224))
    # Convert image to array and expand dimensions to fit the model input shape
    image = np.expand_dims(image, axis=0)
    return image

# Define routes
@app.route("/")
def index():
    # Render index.html template
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the image file from the request
    file = request.files["image"]

    # Read the image file
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    preds = model.predict(processed_image)
    pred_label = lb.classes_[np.argmax(preds)]

    # Render result.html template with the predicted label
    return render_template("result.html", predicted_class=pred_label, image_path="data:image/jpeg;base64," + image_to_base64(image))

def image_to_base64(image):
    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = buffer.tobytes()
    return str(b64encode(jpg_as_text), 'utf-8')

if __name__ == "__main__":
    # Run the application
    app.run(debug=True)
