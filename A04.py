import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchvision.transforms import v2
    HAS_V2 = True
except Exception:
    HAS_V2 = False
    import torchvision.transforms as transforms


def get_approach_names():
    return ["baseline_cnn", "augmented_deeper_cnn"]


def get_approach_description(approach_name):
    descriptions = {
        "baseline_cnn": "A compact CNN with three convolutional blocks and two fully connected layers.",
        "augmented_deeper_cnn": "A deeper CNN with batch normalization, dropout, stronger feature extraction, and training-only data augmentation.",
    }
    return descriptions.get(approach_name, "Unknown approach")


def _get_base_transform():
    if HAS_V2:
        ops = []
        if hasattr(v2, "ToImage"):
            ops.append(v2.ToImage())
            ops.append(v2.ToDtype(torch.float32, scale=True))
        else:
            ops.append(v2.ToImageTensor())
            ops.append(v2.ConvertImageDtype(torch.float32))
        return ops
    return [transforms.ToTensor()]


def get_data_transform(approach_name, training):
    if HAS_V2:
        ops = []

        if training and approach_name == "augmented_deeper_cnn":
            if hasattr(v2, "RandomHorizontalFlip"):
                ops.append(v2.RandomHorizontalFlip(p=0.5))
            if hasattr(v2, "RandomRotation"):
                ops.append(v2.RandomRotation(degrees=12))
            if hasattr(v2, "RandomCrop"):
                ops.append(v2.RandomCrop((32, 32), padding=4))

        ops.extend(_get_base_transform())
        return v2.Compose(ops)

    ops = []
    if training and approach_name == "augmented_deeper_cnn":
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomCrop(32, padding=4),
        ])
    ops.extend(_get_base_transform())
    return transforms.Compose(ops)


def get_batch_size(approach_name):
    if approach_name == "baseline_cnn":
        return 128
    if approach_name == "augmented_deeper_cnn":
        return 96
    return 64


class BaselineCNN(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_cnt),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeeperCNN(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, class_cnt),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(approach_name, class_cnt):
    if approach_name == "baseline_cnn":
        return BaselineCNN(class_cnt)
    if approach_name == "augmented_deeper_cnn":
        return DeeperCNN(class_cnt)
    raise ValueError(f"Unknown approach name: {approach_name}")


def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()

    if approach_name == "baseline_cnn":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        epochs = 12
    elif approach_name == "augmented_deeper_cnn":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        epochs = 18
    else:
        raise ValueError(f"Unknown approach name: {approach_name}")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(epochs // 3, 1),
        gamma=0.5
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / max(test_total, 1)

        print(
            f"[{approach_name}] "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return model