import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "dataset/Training"
test_dir = "dataset/Testing"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_set = datasets.ImageFolder(train_dir, transform=transform)
test_set = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

class HybridModel(nn.Module):
    def _init_(self, num_classes=4):
        super(HybridModel, self)._init_()

        self.backbone = models.swin_v2_t(weights="IMAGENET1K_V1")
        self.backbone.head = nn.Identity()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            swin_out = self.backbone(dummy)
            swin_features = swin_out.shape[1]
            cnn_out = self.cnn(dummy)
            cnn_features = cnn_out.view(1, -1).shape[1]

        self.total_features = swin_features + cnn_features

        self.fc = nn.Sequential(
            nn.Linear(self.total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        swin_out = self.backbone(x)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(x.size(0), -1)
        combined = torch.cat((swin_out, cnn_out), dim=1)
        return self.fc(combined)

model = HybridModel(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 5

os.makedirs("model", exist_ok=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    model_path = f"model/model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")
