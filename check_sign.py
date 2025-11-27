# infer_traffic_signs.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 학습 코드에서 사용한 CLASS_NAMES와 CNN 구조를 동일하게 맞춰야 함
CLASS_NAMES = ["stop", "speedlimit", "trafficlight", "crosswalk"]
NUM_CLASSES = len(CLASS_NAMES)

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        #x = x.view(-1, 128 * 8 * 8)
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(model_path="traffic_sign_cnn.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # 저장된 class_names를 쓰고 싶으면 이렇게:
    saved_class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = CNN(num_classes=len(saved_class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, saved_class_names, device


def predict_image(model, class_names, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #컬러
        #transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    with torch.no_grad():
        outputs = model(x)              # logits
        probs = torch.softmax(outputs, dim=1)[0]  # [num_classes]

    pred_idx = torch.argmax(probs).item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence, probs.cpu().numpy()


if __name__ == "__main__":
    # 1) 모델 불러오기
    model_path = r"D:\Python\simp\traffic_sign_cnn.pth"
    model, class_names, device = load_model(model_path)

    # 2) 테스트할 이미지 경로
    test_image_path = r"D:\Python\simp\check_img\traffic_sign_img2.png"  # <-- 여기 바꿔서 사용

    pred_class, conf, probs = predict_image(model, class_names, device, test_image_path)

    print(f"이미지: {test_image_path}")
    print(f"예측 클래스: {pred_class} (신뢰도: {conf*100:.2f}%)")
    print("각 클래스별 확률:")
    for name, p in zip(class_names, probs):
        print(f"  {name:12s}: {p*100:.2f}%")
