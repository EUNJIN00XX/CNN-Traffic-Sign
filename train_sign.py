import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

##############################
# CLASS 정의
##############################
CLASS_NAMES = ["stop", "speedlimit", "trafficlight", "crosswalk"]
NUM_CLASSES = len(CLASS_NAMES)


class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 폴더 이름 -> 라벨 인덱스 매핑
        # 폴더 이름이 CLASS_NAMES와 동일하다고 가정
        class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        for class_name, label in class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[경고] 폴더 없음: {class_dir}")
                continue

            for img_name in sorted(os.listdir(class_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

        print(f"총 이미지 수: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')  # 흑백

        if self.transform:
            image = self.transform(image)

        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 두 번째 컨볼루션 블록
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 세 번째 컨볼루션 블록
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # 입력이 28x28일 때 feature map 크기: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool2(self.relu2(self.conv2(x)))  # [B, 64, 7, 7]
        x = self.pool3(self.relu3(self.conv3(x)))  # [B, 128, 3, 3]
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 128 * 8 * 8)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x


def plot_history(history, save_path=r"D:\Python\simp\traffic_sign_img\traffic_sign_train.png"):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"학습 그래프 저장: {save_path}")


def main():
    data_dir = r"D:\Python\simp\traffic_sign_img"  # <- 네가 준비한 교통표지판 폴더

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #컬러
        #transforms.Normalize((0.5,), (0.5,)) 흑백
    ])

    full_dataset = TrafficSignDataset(data_dir=data_dir, transform=transform)

    if len(full_dataset) == 0:
        raise RuntimeError("데이터셋이 비어 있습니다. 폴더 구조와 이미지를 확인하세요.")

    # Train / Validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    batch_size = 64
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)

    model = CNN(num_classes=NUM_CLASSES).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss_sum / val_total
        epoch_val_acc = val_correct / val_total
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
        
        # 최고 성능 모델 저장
        model_path = r"D:\Python\simp\traffic_sign_cnn.pth"
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
            }, model_path)
            print(f">>> Best model saved (Val Acc: {epoch_val_acc*100:.2f}%)")

    plot_history(history, save_path=r"D:\Python\simp\train_img\traffic_sign_train.png")
    print(f"최고 검증 정확도: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
