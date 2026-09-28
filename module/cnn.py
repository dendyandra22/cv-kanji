import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# --- 1. Define Lightweight CNN Architecture ---
class KanjiCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(KanjiCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# class KanjiCNN2(nn.Module):
#     def __init__(self, num_classes):
#         super(KanjiCNN2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x))) # [B, 32, 14, 14]
#         x = self.pool(self.relu(self.conv2(x))) # [B, 64, 7, 7]
#         x = x.view(x.size(0), -1)               # Flatten
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # --- 2. SSIM Structural Similarity Evaluator ---
# def evaluate_ssim(user_img_array, template_svg_path):
#     """
#     Computes SSIM between user canvas drawing image and rendered template image.
#     """
#     # Convert RGBA canvas image to grayscale
#     user_gray = cv2.cvtColor(user_img_array, cv2.COLOR_RGBA2GRAY)
#
#     # Invert so background is black (0) and stroke is white (255)
#     _, user_bin = cv2.threshold(user_gray, 200, 255, cv2.THRESH_BINARY_INV)
#
#     # Create dummy reference canvas if template comparison image is not provided
#     # (In production, load/rasterize target Kanji SVG to same size)
#     template_dummy = np.zeros_like(user_bin)
#
#     # Calculate SSIM score
#     score, _ = ssim(user_bin, template_dummy, full=True)
#
#     # Normalize score between 0 and 100
#     ssim_score = max(0.0, (score + 1.0) / 2.0 * 100.0)
#     return round(ssim_score, 2)


# --- 3. CNN Model Prediction Function ---

def preprocess_canvas_image(rgba_array):
    """
    Preprocesses Streamlit canvas drawing to match MNIST/KMNIST format:
    28x28 size, white strokes on black background, normalized.
    """
    # Convert RGBA to Grayscale
    gray = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2GRAY)

    # Threshold canvas: ensure background is 0 (black) and strokes are 255 (white)
    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Bounding Box Cropping (centers the drawn stroke)
    coords = cv2.findNonZero(bin_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding around bounding box
        pad = 20
        y1, y2 = max(0, y - pad), min(bin_img.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(bin_img.shape[1], x + w + pad)
        cropped = bin_img[y1:y2, x1:x2]
    else:
        cropped = bin_img

    # Morphological Dilation (Fattens strokes proportionally to max_dim to avoid fading stroke)
    # The bigger/zoomed-out the bounding box, the more we thicken the stroke before shrinking!
    kernel_size = max(1, int(cropped.shape[0] / 80))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickened = cv2.dilate(cropped, kernel, iterations=1)

    # Resize to 28x28 like KMNIST
    resized = cv2.resize(thickened, (28, 28), interpolation=cv2.INTER_AREA)

    # Transform to PyTorch Tensor & Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Scales pixel values to [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1.0, 1.0]
    ])

    pil_img = Image.fromarray(resized, mode="L")
    tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    return tensor


def predict_kanji_cnn2(model, rgba_array, class_names):

    tensor = preprocess_canvas_image(rgba_array)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    print("predicted_idx", predicted_idx)
    predicted_char = class_names[predicted_idx.item()]
    predicted_char = class_names[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)

    return predicted_char, confidence_pct

def predict_kanji_cnn(pil_image, model, class_names):
    """
    Runs model inference on PIL image and returns target class confidence.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Transform image to PyTorch tensor format (1, 1, 28, 28)
    img_tensor = transform(pil_image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Extract max probability & class index
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_char = class_names[predicted_idx.item()] if class_names else "Unknown"
    confidence_score = round(confidence.item() * 100.0, 2)

    return predicted_char, confidence_score


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SiameseKanjiNet(nn.Module):
#     def __init__(self):
#         super(SiameseKanjiNet, self).__init__()
#         # Shared CNN Feature Extractor
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 14x14
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 7x7
#
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 128)  # Output 128-dimensional embedding vector
#         )
#
#     def forward_one(self, x):
#         return self.encoder(x)
#
#     def forward(self, img1, img2):
#         # Extract feature embeddings for both images
#         emb1 = self.forward_one(img1)
#         emb2 = self.forward_one(img2)
#         return emb1, emb2
#
#
# # Contrastive Loss function to train matching vs non-matching pairs
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, emb1, emb2, label):
#         # label: 0 for same Kanji, 1 for different Kanji
#         euclidean_distance = F.pairwise_distance(emb1, emb2)
#         loss_same = (1 - label) * torch.pow(euclidean_distance, 2)
#         loss_different = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
#         return torch.mean(loss_same + loss_different)
#
#
# import random
# from torch.utils.data import Dataset
#
#
# class KanjiPairDataset(Dataset):
#     def __init__(self, dataset_folder, transform=None):
#         # Load images grouped by folder (e.g. U+4E00, U+4E8C)
#         self.dataset = datasets.ImageFolder(root=dataset_folder)
#         self.transform = transform
#
#         # Group indices by class label
#         self.label_to_indices = {}
#         for idx, (_, label) in enumerate(self.dataset.samples):
#             if label not in self.label_to_indices:
#                 self.label_to_indices[label] = []
#             self.label_to_indices[label].append(idx)
#
#         self.labels = list(self.label_to_indices.keys())
#
#     def __getitem__(self, index):
#         # Pick Image A
#         img1, label1 = self.dataset[index]
#
#         # 50% chance to pair with same class, 50% chance with different class
#         should_match = random.random() > 0.5
#
#         if should_match:
#             idx2 = random.choice(self.label_to_indices[label1])
#             target_label = 0.0  # Same
#         else:
#             label2 = random.choice([l for l in self.labels if l != label1])
#             idx2 = random.choice(self.label_to_indices[label2])
#             target_label = 1.0  # Different
#
#         img2, _ = self.dataset[idx2]
#
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#
#         return img1, img2, torch.tensor(target_label, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.dataset)