import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

from models import AgeGenderModel


class SeniorCitizenDetector:

    def __init__(self):

        import torch

        print("=" * 60)
        print("CUDA Available :", torch.cuda.is_available())
        print("CUDA Version   :", torch.version.cuda)
        print("GPU Count      :", torch.cuda.device_count())

        if torch.cuda.is_available():
            print("GPU Name       :", torch.cuda.get_device_name(0))
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        print("Using Device   :", self.device)
        print("=" * 60)

        # ---------------- YOLO ---------------- #

        self.face_model = YOLO("models/face.pt")
        self.face_model.to(str(self.device))

        # ---------------- EfficientNet ---------------- #

        self.model = AgeGenderModel()

        self.model.load_state_dict(
            torch.load(
                "models/best_model.pth",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        # ---------------- FP16 ---------------- #

        self.fp16 = self.device.type == "cuda"

        if self.fp16:
            self.model.half()

        # ---------------- Transform ---------------- #

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    def detect(self, frame):

        results = self.face_model(
            frame,
            imgsz=320,
            conf=0.45,
            verbose=False,
            device=str(self.device)
        )

        detections = []

        for result in results:

            for box in result.boxes.xyxy.cpu().numpy():

                x1,y1,x2,y2 = map(int,box)

                face = frame[y1:y2,x1:x2]

                if face.size == 0:
                    continue

                rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                img = Image.fromarray(rgb)

                img = self.transform(img)

                img = img.unsqueeze(0).to(self.device)

                if self.fp16:
                    img = img.half()

                with torch.no_grad():

                    age, gender = self.model(img)

                age = int(age.item())

                gender = gender.argmax(1).item()

                gender = "Male" if gender==0 else "Female"

                status = (
                    "Senior Citizen"
                    if age>=60
                    else "Normal"
                )

                color = (
                    (0,0,255)
                    if status=="Senior Citizen"
                    else (0,255,0)
                )

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{gender} {age}",
                    (x1,y1-25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                detections.append({

                    "age":age,

                    "gender":gender,

                    "status":status

                })

        return frame,detections