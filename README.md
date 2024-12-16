# Image-Signature-Validation-Detection-System

This project uses Flask to create a web application that allows users to upload images containing signatures, detect and validate them based on a pre-trained dataset. The system supports uploading images, processes them, and displays the validation result.

# Features

- **Signature Detection**: Detect signatures from uploaded images.
- **Signature Validation**: Match the detected signatures with pre-trained known signatures and validate them.
- **Upload Options**: Users can upload images containing signatures.
- **Output Display**: The processed image is displayed with validation results (real or fake).

# Requirements

Before running the project, make sure you have the following installed:

- Python 3.x
- Flask
- OpenCV
- NumPy
- TensorFlow or any deep learning framework used for model training (if applicable)

# Installation

#  1. Clone the Repository

Clone the repository to your local machine:
git clone https://github.com/YOUR-USERNAME/Image-Sign-Validation-Detection-System.git

# 2. Install Dependencies
Navigate to the project folder and install the necessary dependencies:
cd "C:\Users\Omkar\OneDrive\Documents\Image-Sign-Validation-Detection-System"
pip install -r requirements.txt

# 3. Set Up the Project
Ensure the following directory structure:
Image-Sign-Validation-Detection-System/
├── app.py
├── templates/
│   └── index.html
├── model/ (Processed or trained images model is saved here)
├── dataset/ (Add known signatures for validation here)
├── train_model.py
├── uploads/ (All runtime files uploaded from UI are stored here)
└── requirements.txt

# 4. Add Known Signatures
To train the system, add images of known signatures in the dataset/ folder. For example:

dataset/
├── John Doe/
│   └── john_sign1.jpg
│   └── john_sign2.jpg
├── Jane Smith/
│   └── jane_sign1.jpg
│   └── jane_sign2.jpg

# 5. Running the Application
To run the application, execute:
Train the model with signature data:
python train_model.py
Start the Flask web server:
python app.py
This will start a Flask web server on http://127.0.0.1:5000/.

# Usage
# 1. Accessing the Application
Navigate to http://127.0.0.1:5000/ in your web browser.

# 2. Uploading an Image
Click the Upload an image button and select an image containing a signature from your computer. The system will process the file, detect the signature, and validate it based on the pre-trained data.

# 3. Viewing Processed Output
Once processing is complete, the detected signature will be displayed. If the signature is recognized, it will display the name of the person. If the signature is not recognized, it will display "Unknown".

# 4. Supported File Formats
Images: PNG, JPG, JPEG

# File Structure
app.py: This is the main file that contains the Flask web server. It handles file uploads, signature detection, and validation.

train_model.py: This file contains the logic for encoding signatures from images, saving them, and loading the pre-trained signature encodings. It also includes functions to match detected signatures with known ones.

templates/index.html: The HTML template used by Flask to render the front end. It provides a file upload form and displays the processed image after processing.

# Future Enhancements
Real-time Signature Validation: Implement real-time signature validation for dynamic uploads.

Error Handling: Improve error handling, especially for edge cases (e.g., invalid signatures).

More File Formats: Support additional file formats for images.

Training System: Add functionality to dynamically add new known signatures to the system.
