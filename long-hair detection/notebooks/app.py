import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import os
import streamlit as st
import sys
import timm

st.write("Python Version:", sys.version)
# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Long Hair Identification System",
    layout="wide"
)

st.title("Long Hair Identification System")

st.info("""
Rule Applied:

Age between 20 and 30:
• Long Hair → Female
• No Long Hair → Male

Outside 20-30:
• Use Gender Model Prediction
""")

# ==========================
# DEVICE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==========================
# FILE PATHS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AGE_MODEL_PATH = os.path.join(BASE_DIR, "age_model.pth")
GENDER_MODEL_PATH = os.path.join(BASE_DIR, "gender_model.pth")
HAIR_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# ==========================
# AGE MODEL
# ==========================

@st.cache_resource
def load_age_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=1
    )

    checkpoint = torch.load(
        AGE_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model.to(device)

# ==========================
# GENDER MODEL
# ==========================

@st.cache_resource
def load_gender_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )

    checkpoint = torch.load(
        GENDER_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model.to(device)

# ==========================
# HAIR MODEL
# ==========================

@st.cache_resource
def load_hair_model():

    model = YOLO(HAIR_MODEL_PATH)

    return model

# ==========================
# LOAD MODELS
# ==========================

age_model = load_age_model()
gender_model = load_gender_model()
hair_model = load_hair_model()

# ==========================
# IMAGE TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================
# AGE PREDICTION
# ==========================

def predict_age(image):

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        age = age_model(img)

    return max(0, int(age.item()))

# ==========================
# GENDER PREDICTION
# ==========================

def predict_gender(image):

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = gender_model(img)

    pred = torch.argmax(output, dim=1).item()

    return "Male" if pred == 0 else "Female"

# ==========================
# LONG HAIR DETECTION
# ==========================

def detect_long_hair(image):

    results = hair_model(image)

    for r in results:

        for box in r.boxes:

            cls = int(box.cls.item())

            if cls == 1:
                return True

    return False

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# PREDICTION
# ==========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with st.spinner("Processing..."):

        age = predict_age(image)

        gender = predict_gender(image)

        long_hair = detect_long_hair(image)

        final_gender = gender

        if 20 <= age <= 30:

            if long_hair:
                final_gender = "Female"
            else:
                final_gender = "Male"

    with col2:

        st.subheader("Prediction Results")

        st.write(f"**Predicted Age:** {age}")

        st.write(f"**Gender Model Prediction:** {gender}")

        st.write(f"**Long Hair Detected:** {long_hair}")

        st.success(
            f"Final Gender: {final_gender}"
        )

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.caption(
    "Age Prediction + Gender Classification + Long Hair Detection"
)