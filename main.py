import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
from io import BytesIO

from helpers import CATEGORIES

app = FastAPI()

loaded_model = tf.keras.models.load_model('models/dog_breeds_v1')


def preprocess_image(image_content, target_size):
    """Convert image content to a format suitable for the model."""
    image = Image.open(BytesIO(image_content))
    image = image.resize(target_size)  # Resize the image to the model's input size
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add a batch dimension


@app.post("/breed-detection")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    input_size = (299, 299)
    image_data = preprocess_image(content, input_size)
    predictions = loaded_model.predict(image_data)
    predicted_index = np.argmax(predictions, axis=1)[0]

    return JSONResponse(content={
        "filename": file.filename,
        "content_type": file.content_type,
        "breed": CATEGORIES[predicted_index],
    })


@app.get("/breeds")
async def available_breeds():
    return JSONResponse(content={
        "breeds": CATEGORIES
    })
