from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import io
import logging
from src.utils.utils import load_model_from_file
import os

# Enable logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)


# Path to your trained model
trained_model_file_path = os.path.join("artifacts", "model.h5")
model = load_model_from_file(trained_model_file_path)


# Helper function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the model's expected input size
    image = image.resize((256, 256))  # Assuming the model expects 256x256
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    if request.method == "POST":
        try:
            # Check if the file part is present
            if "file" not in request.files:
                return jsonify({"error": "No file part"}), 400
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            if file:
                # Read and preprocess the image
                image = Image.open(io.BytesIO(file.read()))
                image_array = preprocess_image(image)

                # Make prediction
                predictions = model.predict(image_array)
                predicted_label = (predictions > 0.5).astype("int32")[0][0]

                # Result interpretation
                result = "Tuberculosis" if predicted_label == 1 else "Normal"

                return render_template("result.html", final_result=result)
        except Exception as e:
            app.logger.error(f"Error occurred: {e}")
            return jsonify({"error": str(e)}), 500


# Execution begin
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
