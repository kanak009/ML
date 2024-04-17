import os
from flask import Flask, request, render_template
import joblib
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'george_classifier.pkl'
clf = joblib.load(model_path)

# Define the image upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')

        file = request.files['file']

        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('upload.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform classification
            result = classify_image(filepath)
            return render_template('upload.html', message='File uploaded successfully', result=result)
        else:
            return render_template('upload.html', message='File format not supported')

    return render_template('upload.html', message='Upload an image')

# Function to classify the uploaded image
def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Preprocess the image
    img_resized = cv2.resize(img, (100, 100))  # Resize the image to a fixed size (adjust as needed)
    img_flat = img_resized.flatten()  # Flatten the resized image
    # Perform prediction
    prediction = clf.predict([img_flat])
    return "St. George detected" if prediction[0] == 1 else "No St. George detected"


if __name__ == '__main__':
    app.run(debug=True)
