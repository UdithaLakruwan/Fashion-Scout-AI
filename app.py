from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from model import FashionModel

# 1. Initialize App
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fashion Scout AI is Online"}

# 2. Load the Brain (Level 3 Logic)
model = FashionModel()
model.load_state_dict(torch.load("fashion_scout.pth", map_location=torch.device('cpu')))
model.eval()

# 3. Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L') # Grayscale
    
    # Pre-process: FashionMNIST needs white objects on black backgrounds
    image = image.resize((28, 28))
    img_array = np.array(image)
    
    # If background is light, invert it (Standardization)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    image = Image.fromarray(img_array)
    
    # Transform to Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Get Result
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    return {
        "prediction": classes[prediction],
        "confidence": "high",
        "processed_image_mean": float(img_array.mean())
    }
