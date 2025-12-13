from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- MODEL ----
class VegetableCNN(nn.Module):
    def __init__(self, num_classes=44):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

model = VegetableCNN()
model.load_state_dict(
    torch.load("api/models/fruit_vegetable_model.pth", map_location="cpu")
)
model.eval()

classes = [
    'Bean','Brinjal','Broccoli','Cabbage','Carrot','Cauliflower','Cucumber',
    'Papaya','Potato','Pumpkin','Radish','Tomato','apple','banana','beetroot',
    'bell pepper','bitter gourd','bottle gourd','capsicum','chilli pepper',
    'corn','eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon',
    'lettuce','mango','onion','orange','pear','peas','pineapple','pomegranate',
    'raddish','soy beans','spinach','sweetcorn','sweetpotato','turnip','watermelon'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        img = Image.open(file_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

        _, idx = torch.max(outputs, 1)
        prediction = classes[idx.item()]

        return {
            "status": "success",
            "prediction": prediction
        }

    finally:
        os.remove(file_path)
