from flask import Flask, request, jsonify
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class VegetableCNN(nn.Module):
    def __init__(self, num_classes=44):
        super(VegetableCNN, self).__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
model = VegetableCNN()
model.load_state_dict(torch.load("backend/models/fruit_vegetable_model.pth", map_location="cpu"))
model.eval()

classes =   [
            'Bean', 'Brinjal', 'Broccoli', 'Cabbage', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato', 'apple', 'banana', 'beetroot',
            'bell pepper', 'bitter gourd', 'bottle gourd', 'capsicum', 'chilli pepper', 'corn', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
            'mango', 'onion', 'orange', 'pear', 'peas', 'pineapple', 'pomegranate', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'turnip', 'watermelon'
            ]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image_class(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    image_tensor = transform(img).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image_tensor)
        
    _, predicted_index = torch.max(outputs, 1)
    
    return classes[predicted_index.item()]