import os
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
from io import BytesIO

from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt, JWTError
import httpx

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from helpers import CATEGORIES

app = FastAPI()

# oauth2 config
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# DB config
engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT config
ALGORITHM = "HS256"


class TokenModel(BaseModel):
    token: str


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_model = tf.keras.models.load_model('models/dog_breeds_v1')


class AuthCode(BaseModel):
    code: str


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String, default=False)


Base.metadata.create_all(bind=engine)


class UserInDB(BaseModel):
    id: int
    google_id: str
    email: str
    name: str
    is_admin: bool


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/auth/google")
async def authenticate_user(auth_data: AuthCode, db: Session = Depends(get_db)):
    token_url = "https://oauth2.googleapis.com/token"

    data = {
        "code": auth_data.code,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
        "grant_type": "authorization_code"
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, headers=headers, data=data)
        print("response123: ", response.content)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error exchanging code for token")

    token_data = response.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access token in Google response")

    response = requests.get(f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={access_token}")
    if response.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token")

    user_info = response.json()
    google_id = user_info["sub"]
    email = user_info["email"]
    name = user_info.get("name", "Unknown")

    # Checking whether the user already exists in the database
    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        # Create a new user
        user = User(google_id=google_id, email=email, name=name)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Generating a JWT token for the application
    jwt_token = create_jwt_token({"sub": str(user.id), "email": user.email})
    return {"access_token": jwt_token, "token_type": "bearer"}


def create_jwt_token(data: dict):
    return jwt.encode(data, os.getenv("SECRET_KEY"), algorithm=ALGORITHM)


@app.get("/admin_panel_stats")
async def admin_panel_stats(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return {"email": user.email, "name": user.name}

    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate token")


def breed_images(breed: str, path: str):
    breed_dir = [d for d in os.listdir(path) if d.lower().split("-")[1] == (breed.strip().replace(" ", "_"))]
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
