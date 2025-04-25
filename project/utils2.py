from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from check import IDS, COLS, check_df  # Import everything needed from check.py


CLASS_NAMES = ['Amandina', 'Arabia', 'Comtesse', 'Creme_brulee', 'Jelly_Black', 'Jelly_Milk', 'Jelly_White',
               'Noblesse', 'Noir_authentique', 'Passion_au_lait', 'Stracciatella', 'Tentation_noir', 'Triangolo']

YOLO_NAMES = [
        "Amandina", "Arabia", "Comtesse", "Creme_brulee", "Jelly_Black",
        "Jelly_Milk", "Jelly_White", "Noblesse", "Noir_authentique", "Passion_au_lait",
        "Stracciatella", "Tentation_noir", "Triangolo"
    ]

YOLO_TO_COL = {
        "Amandina": "Amandina",
        "Arabia": "Arabia",
        "Comtesse": "Comtesse",
        "Creme_brulee": "Crème brulée",
        "Jelly_Black": "Jelly Black",
        "Jelly_Milk": "Jelly Milk",
        "Jelly_White": "Jelly White",
        "Noblesse": "Noblesse",
        "Noir_authentique": "Noir authentique",
        "Passion_au_lait": "Passion au lait",
        "Stracciatella": "Stracciatella",
        "Tentation_noir": "Tentation noir",
        "Triangolo": "Triangolo"
    }

################## DATA LOADER ##################
class YoloGridDataset(Dataset):
    
    def __init__(self, image_dir, label_dir,GRID_SIZE, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir.replace('images', 'labels')
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        self.NUM_CLASSES = 13
        self.ANCHORS = 3
        self.GRID_SIZE = GRID_SIZE

    def GetNumClasses():
        return self.NUM_CLASSES
    def GetAnchors():
        return self.ANCHORS
    def GetGridSize():
        return self.GRID_SIZE

    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label_tensor = torch.zeros((self.ANCHORS, self.GRID_SIZE, self.GRID_SIZE, 5 + self.NUM_CLASSES))

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    gx, gy = int(x * self.GRID_SIZE), int(y * self.GRID_SIZE)
                    label_tensor[0, gy, gx, 0:4] = torch.tensor([x, y, w, h])
                    label_tensor[0, gy, gx, 4] = 1
                    label_tensor[0, gy, gx, 5 + int(cls)] = 1

        return img, label_tensor
################################################

########## MODEL 1 WORKS WITH GRID 20 ##########
class TinyYOLO(nn.Module):
    def __init__(self, num_classes=13):
        super(TinyYOLO, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 320x320

            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 160x160

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 80x80

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 40x40

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 20x20

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        )

        # Final detection layer: [B, 20, 20, anchors * (5 + num_classes)]
        self.output = nn.Conv2d(512, 3 * (5 + num_classes), 1)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        # Reshape: (batch, anchors, 5 + num_classes, S, S)
        bs, _, h, w = x.shape
        return x.view(bs, 3, 5 + self.output.out_channels // 3 - 5, h, w)
################################################

########## MODEL 2 WORKS WITH GRID 40 ##########
class CompactYOLOv2(nn.Module):
    def __init__(self, num_classes=13, anchors=3):
        super(CompactYOLOv2, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 320

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 160

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 80

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 40

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.SiLU(),
            nn.MaxPool2d(2, 2),  # 20
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, anchors * (5 + num_classes), 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        bs, _, h, w = x.shape
        return x.view(bs, self.anchors, 5 + self.num_classes, h, w)
################################################

########## MODEL 3 WORKS WITH GRID 40 ##########
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=2):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU()
            ) for _ in range(n)]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * (n + 1), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        outputs = [x1]
        for block in self.blocks:
            x1 = block(x1)
            outputs.append(x1)
        return self.out_conv(torch.cat(outputs, dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        self.cv1 = nn.Conv2d(in_channels, out_channels // 2, 1, 1)
        self.cv2 = nn.Conv2d(out_channels // 2 * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

class YOLOv8Lite(nn.Module):
    def __init__(self, num_classes=13, anchors=3):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        def cbs(in_ch, out_ch, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )

        self.stem = nn.Sequential(
            cbs(3, 64, 3, 2, 1),
            C2f(64, 128),
            cbs(128, 128, 3, 2, 1),
            C2f(128, 256),
            cbs(256, 256, 3, 2, 1),
            C2f(256, 512),
            cbs(512, 512, 3, 2, 1),
            C2f(512, 512),
        )

        self.neck = nn.Sequential(
            SPPF(512, 512),
            cbs(512, 512, 3, 1, 1)
        )

        self.head = nn.Conv2d(512, anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.neck(x)
        x = self.head(x)
        bs, _, h, w = x.shape
        return x.view(bs, self.anchors, 5 + self.num_classes, h, w)
################################################

# TOTAL NUMBER OF PARAMETER PRINTER #
def Print_Total_Number_Of_Parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
################################################

################### TRAINING ###################
# Improved loss function
def yolo_loss(pred, target):
    # Reshape to match target
    pred = pred.permute(0, 1, 3, 4, 2)  # [B, A, S, S, 5+C]
    
    # Components
    pred_box = pred[..., 0:4]
    pred_obj = pred[..., 4]
    pred_cls = pred[..., 5:]

    true_box = target[..., 0:4]
    true_obj = target[..., 4]
    true_cls = target[..., 5:]

    # Coordinate loss (only where object exists)
    coord_loss = F.mse_loss(pred_box[true_obj == 1], true_box[true_obj == 1])

    # Objectness loss
    obj_loss = F.binary_cross_entropy_with_logits(pred_obj, true_obj)

    # Classification loss
    cls_loss = F.binary_cross_entropy_with_logits(pred_cls[true_obj == 1], true_cls[true_obj == 1])

    return coord_loss + obj_loss + cls_loss

def Train_Model(model: nn.Module,EPOCHS: int, optimizer: Optimizer, device: torch.device, per_epoch_save: int, loader: DataLoader ):

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = yolo_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
    
        if (epoch + 1) % per_epoch_save == 0 or (epoch + 1) == EPOCHS:
            model_path = f"models/YOLOv8Lite_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved to: {model_path}")
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {total_loss:.4f}")

    return model
################################################

######### APPLYING MODEL ON SOME TEST DATA ######
# without NMS
def Use_Model_On_Images(number_of_images: int, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2):

    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    NUM_CLASSES = IMG_PARAM["NUM_CLASSES"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    # === Image Transform ===
    transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Get test images
    test_dir = "test_annotated/images"
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    sampled_paths = np.random.choice(image_paths, size=number_of_images, replace=False).tolist()

    for img_path in sampled_paths:
        img_orig = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_orig.size

        img_resized = img_orig.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()

        boxes = []
        for anchor in range(ANCHORS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    obj_score = torch.sigmoid(torch.tensor(preds[anchor, gy, gx, 4])).item()
                    if obj_score > conf_threshold:
                        x, y, bw, bh = preds[anchor, gy, gx, 0:4]
                        x = (gx + torch.sigmoid(torch.tensor(x)).item()) / GRID_SIZE
                        y = (gy + torch.sigmoid(torch.tensor(y)).item()) / GRID_SIZE
                        bw = bw ** 2
                        bh = bh ** 2
                        class_probs = torch.softmax(torch.tensor(preds[anchor, gy, gx, 5:]), dim=0)
                        cls_id = torch.argmax(class_probs).item()
                        cls_conf = class_probs[cls_id].item()

                        boxes.append([x, y, bw, bh, obj_score, cls_id])

        # ✅ Convert RGB to BGR before drawing
        img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

        for box in boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)
        
            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Convert to RGB for plotting
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title(f"Prediction (Resized): {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

# with NMS
def Use_Model_On_Images_NMS(number_of_images: int, model: torch.nn.Module, IMG_PARAM: dict,conf_threshold: float = 0.13, min_dist: float = 40.0):


    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    NUM_CLASSES = IMG_PARAM["NUM_CLASSES"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    # === Image Transform ===
    transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dir = "test_annotated/images"
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    sampled_paths = np.random.choice(image_paths, size=number_of_images, replace=False).tolist()

    for img_path in sampled_paths:
        img_orig = Image.open(img_path).convert("RGB")
        img_resized = img_orig.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()

        # Collect all boxes first
        all_boxes = []
        for anchor in range(ANCHORS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    obj_score = torch.sigmoid(torch.tensor(preds[anchor, gy, gx, 4])).item()
                    if obj_score > conf_threshold:
                        x, y, bw, bh = preds[anchor, gy, gx, 0:4]
                        x = (gx + torch.sigmoid(torch.tensor(x)).item()) / GRID_SIZE
                        y = (gy + torch.sigmoid(torch.tensor(y)).item()) / GRID_SIZE
                        bw = bw ** 2
                        bh = bh ** 2
                        class_probs = torch.softmax(torch.tensor(preds[anchor, gy, gx, 5:]), dim=0)
                        cls_id = torch.argmax(class_probs).item()
                        cls_conf = class_probs[cls_id].item()

                        all_boxes.append([x, y, bw, bh, obj_score, cls_id])

        # Suppress close boxes based on min_dist
        kept_boxes = []
        for box in sorted(all_boxes, key=lambda b: b[4], reverse=True):  # sort by confidence
            bx, by = box[0] * IMG_SIZE, box[1] * IMG_SIZE
            too_close = False
            for kept in kept_boxes:
                kx, ky = kept[0] * IMG_SIZE, kept[1] * IMG_SIZE
                if np.hypot(bx - kx, by - ky) < min_dist:
                    too_close = True
                    break
            if not too_close:
                kept_boxes.append(box)

        # Draw boxes
        img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        for box in kept_boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)

            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show result
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title(f"Prediction (Filtered): {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

################ HELPER ################
def find_image_path_by_partial_name(test_dir, partial_name):
    for f in os.listdir(test_dir):
        if partial_name.split(".")[0] in f and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(test_dir, f)
    return None
##------------------------------------##

# with list of specific names without NMS
def Use_Model_On_Images_by_Name(image_names: list, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Image and model parameters
    test_dir = "test_annotated/images"
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]

    
    # === Image Transform ===
    transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
    ])


    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])

    for name in image_names:
        img_path = find_image_path_by_partial_name(test_dir, name)

        if img_path is None:
            print(f"⚠️ No image found containing: {name}")
            continue

        img_orig = Image.open(img_path).convert("RGB")
        img_resized = img_orig.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()

        boxes = []
        for anchor in range(ANCHORS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    obj_score = torch.sigmoid(torch.tensor(preds[anchor, gy, gx, 4])).item()
                    if obj_score > conf_threshold:
                        x, y, bw, bh = preds[anchor, gy, gx, 0:4]
                        x = (gx + torch.sigmoid(torch.tensor(x)).item()) / GRID_SIZE
                        y = (gy + torch.sigmoid(torch.tensor(y)).item()) / GRID_SIZE
                        bw = bw ** 2
                        bh = bh ** 2
                        class_probs = torch.softmax(torch.tensor(preds[anchor, gy, gx, 5:]), dim=0)
                        cls_id = torch.argmax(class_probs).item()
                        cls_conf = class_probs[cls_id].item()

                        boxes.append([x, y, bw, bh, obj_score, cls_id])

        # Draw boxes on image
        img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        for box in boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)

            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display result
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title(f"Prediction: {name}")
        plt.axis('off')
        plt.show()




# with list of specific names with NMS
def Use_Model_On_Images_by_Name_NMS(image_names: list, model: torch.nn.Module,  IMG_PARAM: dict, conf_threshold: float = 0.2, min_dist: float = 30.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dir = "test_annotated/images"
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]


    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])

    for name in image_names:
        img_path = find_image_path_by_partial_name(test_dir, name)

        if img_path is None:
            print(f"⚠️ No image found containing: {name}")
            continue

        img_orig = Image.open(img_path).convert("RGB")
        img_resized = img_orig.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()

        all_boxes = []
        for anchor in range(ANCHORS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    obj_score = torch.sigmoid(torch.tensor(preds[anchor, gy, gx, 4])).item()
                    if obj_score > conf_threshold:
                        x, y, bw, bh = preds[anchor, gy, gx, 0:4]
                        x = (gx + torch.sigmoid(torch.tensor(x)).item()) / GRID_SIZE
                        y = (gy + torch.sigmoid(torch.tensor(y)).item()) / GRID_SIZE
                        bw = bw ** 2
                        bh = bh ** 2
                        class_probs = torch.softmax(torch.tensor(preds[anchor, gy, gx, 5:]), dim=0)
                        cls_id = torch.argmax(class_probs).item()
                        class_conf = class_probs[cls_id].item()

                        all_boxes.append([x, y, bw, bh, obj_score, cls_id])

        # Distance-based suppression (greedy NMS-like)
        kept_boxes = []
        for box in sorted(all_boxes, key=lambda b: b[4], reverse=True):
            bx, by = box[0] * IMG_SIZE, box[1] * IMG_SIZE
            too_close = False
            for kept in kept_boxes:
                kx, ky = kept[0] * IMG_SIZE, kept[1] * IMG_SIZE
                if np.hypot(bx - kx, by - ky) < min_dist:
                    too_close = True
                    break
            if not too_close:
                kept_boxes.append(box)

        # Draw results
        img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        for box in kept_boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)

            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title(f"Prediction (Filtered): {name}")
        plt.axis('off')
        plt.show()
################################################

################ TABLE CREATION ################
def CREATE_TABLE_FROM_PT(model_path: str, model_class: torch.nn.Module,IMG_PARAM, conf_threshold: float = 0.2):
    COLS = list(YOLO_TO_COL.values())

    NUM_CLASSES = len(YOLO_NAMES)
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image paths
    test_dir = "test_annotated/images"
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    all_rows = []

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()

        counts = {col: 0 for col in COLS}

        for anchor in range(ANCHORS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    obj_score = torch.sigmoid(torch.tensor(preds[anchor, gy, gx, 4])).item()
                    if obj_score > conf_threshold:
                        class_probs = torch.softmax(torch.tensor(preds[anchor, gy, gx, 5:]), dim=0)
                        cls_id = torch.argmax(class_probs).item()
                        class_name = YOLO_NAMES[cls_id]
                        column_name = YOLO_TO_COL[class_name]
                        counts[column_name] += 1
        # Parse file ID and add row
        filename = os.path.basename(img_path)
        
        # Find the index of 'L' and take the next 7 characters
        if 'L' in filename:
            start = filename.index('L') + 1
            file_id_str = filename[start:start+7]
            try:
                file_id = int(file_id_str)
            except ValueError:
                print(f"⚠️ Not a valid 7-digit numeric ID: {file_id_str}")
                continue
        else:
            print(f"⚠️ 'L' not found in filename: {filename}")
            continue
        
        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)

        if (idx + 1) % 10 == 0:
            print(f"{(idx + 1) / len(image_paths) * 100:.2f}% done")

    df = pd.DataFrame(all_rows)
    df = df.set_index("id").sort_index().reset_index()
    df[COLS] = df[COLS].astype(int)
    
    check_df(df, df_name="Generated Submission")
    output_path = f"submission_conf{int(conf_threshold * 100)}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved formatted CSV to: {output_path}")

    return df

def CREATE_TABLE_FROM_PT_NMS(model_path: str, model_class: torch.nn.Module,IMG_PARAM, conf_threshold: float = 0.2, min_dist: float = 30.0):
    COLS = list(YOLO_TO_COL.values())

    NUM_CLASSES = len(YOLO_NAMES)
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load images
    test_dir = "test_annotated/images"
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    all_rows = []

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()

        all_boxes = []

        for anchor in range(ANCHORS):
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    obj_score = torch.sigmoid(torch.tensor(preds[anchor, gy, gx, 4])).item()
                    if obj_score > conf_threshold:
                        x, y, bw, bh = preds[anchor, gy, gx, 0:4]
                        x = (gx + torch.sigmoid(torch.tensor(x)).item()) / GRID_SIZE
                        y = (gy + torch.sigmoid(torch.tensor(y)).item()) / GRID_SIZE
                        bw = bw ** 2
                        bh = bh ** 2
                        class_probs = torch.softmax(torch.tensor(preds[anchor, gy, gx, 5:]), dim=0)
                        cls_id = torch.argmax(class_probs).item()
                        cls_conf = class_probs[cls_id].item()
                        all_boxes.append([x, y, bw, bh, obj_score, cls_id])

        # Apply distance-based filtering
        kept_boxes = []
        for box in sorted(all_boxes, key=lambda b: b[4], reverse=True):  # sort by confidence
            bx, by = box[0] * IMG_SIZE, box[1] * IMG_SIZE
            too_close = False
            for kept in kept_boxes:
                kx, ky = kept[0] * IMG_SIZE, kept[1] * IMG_SIZE
                if np.hypot(bx - kx, by - ky) < min_dist:
                    too_close = True
                    break
            if not too_close:
                kept_boxes.append(box)

        # Count predictions
        counts = {col: 0 for col in COLS}
        for box in kept_boxes:
            cls_id = box[5]
            class_name = YOLO_NAMES[cls_id]
            column_name = YOLO_TO_COL[class_name]
            counts[column_name] += 1

        # Parse file ID and add row
        filename = os.path.basename(img_path)

        # Find the index of 'L' and take the next 7 characters
        if 'L' in filename:
            start = filename.index('L') + 1
            file_id_str = filename[start:start+7]
            try:
                file_id = int(file_id_str)
            except ValueError:
                print(f"⚠️ Not a valid 7-digit numeric ID: {file_id_str}")
                continue
        else:
            print(f"⚠️ 'L' not found in filename: {filename}")
            continue
        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)

        if (idx + 1) % 10 == 0:
            print(f"{(idx + 1) / len(image_paths) * 100:.2f}% done")

    df = pd.DataFrame(all_rows)
    df = df.set_index("id").sort_index().reset_index()
    df[COLS] = df[COLS].astype(int)

    # Save file
    check_df(df, df_name="Generated Submission")
    output_path = f"submission_conf{int(conf_threshold * 100)}_dist{int(min_dist)}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved filtered CSV to: {output_path}")
    return df
################################################




############## CREATE CONTROLLER AND LOCAL SCORE CALCULATION ##############

def create_controller_csv(label_dir='test_annotated/labels', output_path='DoNotSubmitThis_Controller/controller.csv'):


    COLS = ['id'] + list(YOLO_TO_COL.values())
    all_rows = []

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        try:
            # Extract numeric ID from file name
            file_id = int(fname.split('_')[0].lstrip('L'))
        except ValueError:
            print(f"⚠️ Skipping malformed filename: {fname}")
            continue

        # Initialize counts
        counts = {col: 0 for col in YOLO_TO_COL.values()}

        with open(os.path.join(label_dir, fname), 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_name = CLASS_NAMES[class_id]
                    col_name = YOLO_TO_COL[class_name]
                    counts[col_name] += 1

        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Ensure all expected columns exist (especially when some classes are missing)
    for col in COLS[1:]:
        if col not in df.columns:
            df[col] = 0

    df = df[COLS]
    df[COLS[1:]] = df[COLS[1:]].astype(int)  # force integer type
    df = df.set_index("id").sort_index().reset_index()

    df.to_csv(output_path, index=False)
    print(f"✅ Saved controller CSV to: {output_path}")

    check_df(df, df_name="Generated Submission")
    return df


def evaluate_f1_score(pred_df: pd.DataFrame, controller_path: str = "DoNotSubmitThis_Controller/controller.csv") -> float:
    # Load ground truth
    gt_df = pd.read_csv(controller_path)

    # Align both DataFrames by 'id'
    df = pd.merge(gt_df, pred_df, on='id', suffixes=('_true', '_pred'))
    df = df.sort_values('id').reset_index(drop=True)

    class_names = [col for col in gt_df.columns if col != "id"]
    print(class_names)

    f1_scores = []
    for _, row in df.iterrows():
        TP = 0
        FPN = 0
        for cls in class_names:
            y_true = row[f"{cls}_true"]
            y_pred = row[f"{cls}_pred"]
            TP += min(y_true, y_pred)
            FPN += abs(y_true - y_pred)
        f1 = 2 * TP / (2 * TP + FPN) if (2 * TP + FPN) > 0 else 0
        f1_scores.append(f1)

    mean_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n✅ Mean F1 Score: {mean_f1:.4f}")
    return mean_f1




############################################################################