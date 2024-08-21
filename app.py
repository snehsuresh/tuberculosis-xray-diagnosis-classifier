from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import io
import logging
from src.utils.utils import load_model_from_file
import os
import base64

# Enable logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Path to your trained model
trained_model_file_path = os.path.join("artifacts", "model.h5")


# Helper function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the model's expected input size
    image = image.resize((128, 128))  # Model expects 128x128
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0

    # If the model expects a single channel, ensure that the image array has only one channel
    if image_array.ndim == 3 and image_array.shape[-1] == 3:  # Convert RGB to grayscale
        image_array = np.mean(image_array, axis=-1, keepdims=True)

    # Add batch dimension (make it (1, 128, 128, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    model = load_model_from_file(trained_model_file_path)
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
                image = Image.open(io.BytesIO(file.read())).convert(
                    "RGB"
                )  # Ensure image is in RGB mode
                image_array = preprocess_image(image)

                # Make prediction
                predictions = model.predict(image_array)
                predicted_label = (predictions > 0.5).astype("int32")[0][0]

                # Result interpretation
                result = "Tuberculosis" if predicted_label == 1 else "Normal"
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buffered.getvalue()).decode()
                )
                return render_template(
                    "result.html",
                    final_result=result,
                    image_url=img_str,
                )
        except Exception as e:
            app.logger.error(f"Error occurred: {e}")
            return jsonify({"error": str(e)}), 500


# Execution begin
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
