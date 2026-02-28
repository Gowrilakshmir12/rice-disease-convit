import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
def reshape_transform(tensor, height=14, width=14):
    tensor = tensor[:, 1:, :]  # remove class token
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f8f4;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------
st.markdown("<h1 style='text-align: center;'>🌾 Rice Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered ConViT Model for Farmers</p>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------
# CLASS LABELS (MATCH TRAINING ORDER)
# ---------------------------------------------------
CLASS_NAMES = [
    "Rice Blast",
    "Brown Spot",
    "Healthy",
    "Rice Hispa",
    "Rice Scald"
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
target_layers = [model.blocks[-1].norm1]
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)
# ---------------------------------------------------
# IMAGE TRANSFORMATION
# ---------------------------------------------------
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📷 Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform_image(image)
    model.zero_grad()

    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

    confidence, predicted_class = torch.max(probabilities, 0)
    # -------------------------------
# GRAD-CAM SECTION
# -------------------------------
    targets = [ClassifierOutputTarget(predicted_class.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]
    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item() * 100
    # Resize original image to 224x224 (same as model input)
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # ---------------------------------------------------
    # RESULTS SECTION
    # ---------------------------------------------------
    with col2:
        st.markdown("### 🦠 Prediction Result")
        
        st.markdown(f"""
            <div class="result-box">
                <h2 style='color:#2e7d32;'>{predicted_label}</h2>
                <h4>Confidence: {confidence_score:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence_score))

    st.divider()
    st.subheader("🔍 Model Attention (Grad-CAM)")

    st.image(
        visualization,
        caption="Highlighted regions the model focused on",
        use_column_width=True
    )

    # ---------------------------------------------------
    # TOP 3 PREDICTIONS
    # ---------------------------------------------------
    st.subheader("📊 Top 3 Predictions")

    probs_df = pd.DataFrame({
        "Disease": CLASS_NAMES,
        "Probability (%)": probabilities.detach().cpu().numpy() * 100
    })

    probs_df = probs_df.sort_values(by="Probability (%)", ascending=False)

    st.bar_chart(probs_df.set_index("Disease"))

    st.dataframe(probs_df.head(3), use_container_width=True)

    # ---------------------------------------------------
    # HEALTH ADVICE SECTION
    # ---------------------------------------------------
    st.subheader("🌱 Suggested Action")

    if predicted_label == "Healthy":
        st.success("Your crop looks healthy. Keep monitoring regularly.")
    else:
        st.warning("Disease detected. Consider consulting an agricultural expert or applying appropriate treatment.")