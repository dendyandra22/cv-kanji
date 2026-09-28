# # run on kaggle/colab for GPU
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from static_evaluator import KanjiCNN
#
# class KanjiCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(KanjiCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         return self.classifier(x)
#
# # 1. Load KMNIST Dataset (PyTorch downloads this automatically)
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # 2. Train Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = KanjiCNN(num_classes=10).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# print("Training Kanji CNN for 3 epochs...")
# model.train()
# for epoch in range(3):
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#     print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f}")
#
# # 3. Save Model Weights
# torch.save(model.state_dict(), "kanji_cnn.pth")
# print("Saved trained model weights to kanji_cnn.pth!")

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


# ==========================================
# 1. Siamese Network Architecture
# ==========================================
class SiameseKanjiNet(nn.Module):
    def __init__(self):
        super(SiameseKanjiNet, self).__init__()
        # Shared CNN Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: [32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: [64, 7, 7]

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128)  # 128-dimensional embedding space
        )

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        return emb1, emb2


# ==========================================
# 2. Contrastive Loss Function
# ==========================================
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Pushes representations of different classes apart (margin)
    and pulls representations of the same class together.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # label: 0.0 for Same Kanji (Positive Pair), 1.0 for Different Kanji (Negative Pair)
        euclidean_distance = F.pairwise_distance(emb1, emb2)

        loss_same = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_diff = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return torch.mean(loss_same + loss_diff)


# ==========================================
# 3. Dynamic Image Pair Dataset
# ==========================================
class KanjiPairDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.raw_dataset = datasets.ImageFolder(root=dataset_folder)
        self.transform = transform

        # Group sample indices by their label (folder)
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(self.raw_dataset.samples):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __getitem__(self, index):
        # Image A (Anchor)
        path1, label1 = self.raw_dataset.samples[index]
        img1 = Image.open(path1).convert("L")

        # 50% chance for Same class (Positive), 50% chance for Different class (Negative)
        should_match = random.random() > 0.5

        if should_match:
            idx2 = random.choice(self.label_to_indices[label1])
            target_label = 0.0  # 0 = Same
        else:
            different_label = random.choice([l for l in self.labels if l != label1])
            idx2 = random.choice(self.label_to_indices[different_label])
            target_label = 1.0  # 1 = Different

        path2, _ = self.raw_dataset.samples[idx2]
        img2 = Image.open(path2).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(target_label, dtype=torch.float32)

    def __len__(self):
        return len(self.raw_dataset.samples)


# ==========================================
# 4. Main Training Loop
# ==========================================
def train():
    DATASET_DIR = "data/kanjivg_genImg"  # Directory generated by build_kanjivg_dataset
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found!")
        return

    # Image Transforms (Grayscale, 28x28, Tensor, [-1, 1] Normalization)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("📁 Preparing Pair Dataset...")
    pair_dataset = KanjiPairDataset(dataset_folder=DATASET_DIR, transform=transform)
    train_loader = DataLoader(
        pair_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Training on device: {device}")

    model = SiameseKanjiNet().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Starting Siamese training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] ── Loss: {avg_loss:.4f}")

    # Save Model Weights
    torch.save(model.state_dict(), "data/siamese_kanji.pth")
    print("\n✅ Trained weights saved successfully to 'data/siamese_kanji.pth'!")


if __name__ == "__main__":
    train()