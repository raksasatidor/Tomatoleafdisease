import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import os

# Load the trained model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)  # Adjust based on the number of classes
model.load_state_dict(torch.load(r"C:\Users\Guardino\Desktop\PROJECT AI NUSANTARA\tomato3_leaf_disease_model.pth", map_location=torch.device('cpu')))
model.eval()




# Define the class names (adjust based on your dataset)
class_names = ['Early Blight', 'Healthy', 'Late Blight', 'Septoria leaf spot', 'Tomato mosaic virus']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Tomato Leaf Disease Detection")
st.write("Upload an image of a tomato leaf to predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Transform the image and make a prediction
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = class_names[predicted]

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence Score:** {confidence.item():.2f}")

