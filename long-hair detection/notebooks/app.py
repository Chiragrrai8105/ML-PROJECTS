import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


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

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

@st.cache_resource
def load_age_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=1
    )

    model.load_state_dict(
        torch.load(
            "age_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model.to(device)


@st.cache_resource
def load_gender_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )

    model.load_state_dict(
        torch.load(
            "gender_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model.to(device)


@st.cache_resource
def load_hair_model():

    model = YOLO("best.pt")

    return model


age_model = load_age_model()
gender_model = load_gender_model()
hair_model = load_hair_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_age(image):

    img = transform(image)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        age = age_model(img)

    return max(0, int(age.item()))


def predict_gender(image):

    img = transform(image)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        output = gender_model(img)

    pred = torch.argmax(
        output,
        dim=1
    ).item()

    return "Male" if pred == 0 else "Female"


def detect_long_hair(image):

    results = hair_model(image)

    for r in results:

        for box in r.boxes:

            cls = int(box.cls)

            if cls == 1:
                return True

    return False


uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

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

        # Project Logic

        final_gender = gender

        if 20 <= age <= 30:

            if long_hair:
                final_gender = "Female"
            else:
                final_gender = "Male"

    with col2:

        st.subheader("Prediction Results")

        st.write(f"**Predicted Age:** {age}")

        st.write(f"**Long Hair Detected:** {long_hair}")

        st.success(
            f"Resulted Gender: {final_gender}"
        )

st.markdown("---")
