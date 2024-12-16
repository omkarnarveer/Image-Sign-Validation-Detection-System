import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template ,send_from_directory
#from flask import Flask, request, jsonify, render_template ,
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = r'C:\Users\Omkar\OneDrive\Documents\Image Signature Validation System\uploads'
TEMPLATES_FOLDER = r'C:\Users\Omkar\OneDrive\Documents\Image Signature Validation System'
#TEMPLATES_FOLDER = r'C:\Users\Omkar\OneDrive\Documents\Image Signature Validation System\templates'
MODEL_FOLDER = r'C:\Users\Omkar\OneDrive\Documents\Image Signature Validation System\model'
DATASET_FOLDER = r'C:\Users\Omkar\OneDrive\Documents\Image Signature Validation System\dataset'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = load_model(os.path.join(MODEL_FOLDER, 'signature_recognition_model.keras'))

# Load the class indices from the model to map the predicted index to a person's name
class_indices = {v: k for k, v in enumerate(os.listdir(DATASET_FOLDER))}  # Automatically generate class to person mapping
print("Class Indices: ", class_indices)  # Debugging line

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for index page
@app.route('/')
def index():
    #return render_template('index.html')  # Ensure you have an index.html in templates folder
    return send_from_directory('.', 'index.html')

# Route for file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Read image and preprocess
            img = cv2.imread(filepath)

            if img is None:
                return jsonify({'error': 'Invalid image file'})

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the image to match model input size (224x224)
            img = cv2.resize(img, (224, 224))

            # Convert grayscale image back to RGB (3 channels)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Prepare the image for prediction
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image

            # Predict with the model
            prediction = model.predict(img_array)
            print("Model Prediction Output: ", prediction)  # Debugging line

            # Get the index of the predicted person
            predicted_index = np.argmax(prediction)
            print("Predicted Index: ", predicted_index)  # Debugging line

            # Map the predicted index to the person name
            predicted_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_index)]
            print("Predicted Name: ", predicted_name)  # Debugging line

            # Extract the person's name from the folder (which is the name of the dataset folder)
            expected_name = filename.split('_')[0]  # Assuming image filename contains person's name (e.g., 'Omkar_image.jpg')
            print("Expected Name: ", expected_name)  # Debugging line

            # Determine if the signature is Real or Fake
            if predicted_name.lower() == expected_name.lower():
                is_real = "Real"
            else:
                is_real = "Fake"

            # Construct the response message
            response_message = f"Predicted Sign is {is_real} of {predicted_name}"

            # Return the predicted message
            return jsonify({'result': response_message})
        else:
            return jsonify({'error': 'Invalid file format'})

    except Exception as e:
        print("Error: ", str(e))  # Debugging line
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
