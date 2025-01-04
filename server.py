import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
from io import BytesIO

from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt, JWTError
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from helpers import CATEGORIES

app = FastAPI()

# oauth2 config
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# DB config
engine = create_async_engine(os.getenv("DATABASE_URL"))
AsyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession,
                                 expire_on_commit=False)
Base = declarative_base()

# JWT config
ALGORITHM = "HS256"


class TokenModel(BaseModel):
    token: str


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # os.getenv("CORS_ORIGINS", "").split(","),
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


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


create_tables()


class UserInDB(BaseModel):
    id: int
    google_id: str
    email: str
    name: str
    is_admin: bool


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


def is_on_production():
    return os.getenv("DEBUG").lower().strip() != "true"


@app.post("/auth/google")
async def authenticate_user(response: Response, auth_data: AuthCode, db: AsyncSession = Depends(get_db)):
    token_url = "https://oauth2.googleapis.com/token"
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

    # server config validation
    if not (client_id and client_secret and redirect_uri):
        raise HTTPException(status_code=500, detail="Server environment configuration error")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "code": auth_data.code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }

    try:
        async with httpx.AsyncClient() as client:
            # exchange of a code for a token
            response2 = await client.post(token_url, headers=headers, data=data)
            response2.raise_for_status()
            token_data = response2.json()

            access_token = token_data.get("access_token")
            if not access_token:
                raise HTTPException(status_code=400, detail="No access token in Google response")

            # Token verification
            response3 = await client.get(f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={access_token}")
            response3.raise_for_status()
            user_info = response3.json()

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"HTTP request error: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {e.response.text}")

    # user data procession
    google_id = user_info.get("sub")
    email = user_info.get("email")
    name = user_info.get("name", "Unknown")

    if not (google_id and email):
        raise HTTPException(status_code=400, detail="Invalid user data from Google")

    # Downloading or creating a user in the database
    user = await get_user_by_google_id(db, google_id)
    if not user:
        user = User(google_id=google_id, email=email, name=name)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # Create JWT
    jwt_token = create_jwt_token({"sub": str(user.id), "email": user.email})
    secure_flag = is_on_production()

    # create a cookie
    response.set_cookie(
        key="access_token",
        value=jwt_token,
        httponly=True,
        secure=secure_flag,
        samesite="Strict" if is_on_production() else "Lax",
        domain=None,
        path="/",
    )

    return {"message": "Login successful"}


def create_jwt_token(data: dict):
    return jwt.encode(data, os.getenv("SECRET_KEY"), algorithm=ALGORITHM)


async def get_user_by_id(session: AsyncSession, user_id: int):
    async with session.begin():
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        return result.scalars().first()


async def get_user_by_google_id(session: AsyncSession, user_google_id: int):
    async with session.begin():
        stmt = select(User).where(User.google_id == user_google_id)
        result = await session.execute(stmt)
        user = result.scalars().first()
        return user


@app.get("/admin_panel_stats")
async def admin_panel_stats(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        user = await get_user_by_id(db, user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return {"email": user.email, "name": user.name}

    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate token")


@app.get("/auth/verify")
def verify_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return {"authenticated": False}

    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[ALGORITHM])
        return {"authenticated": True, "user": payload["sub"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=is_on_production(),
        samesite="Strict" if is_on_production() else "Lax",
    )
    return {"message": "Logged out successfully"}


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
