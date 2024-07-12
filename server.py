import os

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
from io import BytesIO

from helpers import CATEGORIES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_model = tf.keras.models.load_model('models/dog_breeds_v1')


def breed_images(breed: str, path: str):
    breed_dir = [d for d in os.listdir(path) if d.lower().endswith(breed.strip().replace(" ", "-"))]
    if breed_dir:
        images = [os.path.join(path, breed_dir[0], f) for f in os.listdir(os.path.join(path, breed_dir[0])) if
                  os.path.isfile(os.path.join(path, breed_dir[0], f)) and f.endswith('.jpg')]
        return images


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
        "breed": CATEGORIES[predicted_index].replace("_", " "),
    })


@app.get("/breeds")
async def available_breeds():
    return JSONResponse(content={
        "breeds": CATEGORIES
    })


@app.get("/image")
async def get_image(breed_name: str):
    path = "Images"
    imgs = breed_images(breed_name, path)
    if imgs:
        return FileResponse(imgs[0])
    raise HTTPException(status_code=404, detail="Breed not found")
