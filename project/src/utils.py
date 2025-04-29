from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm
import numpy as np
from .lossfunctions import yolo_loss
import cv2
from typing import Union, List
import re
import pandas as pd
from check import check_df, COLS, IDS, TermFormat

try:
    from torchsummary import summary
    def print_model_summary(model: nn.Module, test_input=(1, 3, 640, 640)):
        print(summary(model, test_input))
except ImportError:
    pass

class YoloGridDataset(Dataset):
    def __init__(self, root_dir, GRID_SIZE, image_size=(640, 640)):
        self.image_dir = Path(f"{root_dir}/images")
        self.label_dir = Path(f"{root_dir}/labels")
        self.image_size = image_size
        self.img_exts = ('.jpg', '.jpeg', '.png')
        self.images = [f for f in self.image_dir.iterdir() if f.suffix.lower() in self.img_exts]
        self.labels = [f for f in self.label_dir.iterdir() if f.suffix.lower() == '.txt']

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor()
        ])
        
        self.NUM_CLASSES = 13
        self.ANCHORS = 3
        self.GRID_SIZE = GRID_SIZE

    def GetNumClasses(self):
        return self.NUM_CLASSES
    def GetAnchors(self):
        return self.ANCHORS
    def GetGridSize(self):
        return self.GRID_SIZE

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

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

def yolo_dataloader(root_dir, GRID_SIZE, image_size=(640, 640), batch_size=4, shuffle=True, collate_fn=None):
    
    dataset = YoloGridDataset(root_dir=root_dir, GRID_SIZE=GRID_SIZE, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_class_names():
    return ['Amandina', 'Arabia', 'Comtesse', 'Creme_brulee', 'Jelly_Black', 'Jelly_Milk', 'Jelly_White', 'Noblesse', 'Noir_authentique', 'Passion_au_lait', 'Stracciatella', 'Tentation_noir', 'Triangolo']

def get_yolo_names():
    return ["Amandina", "Arabia", "Comtesse", "Creme_brulee", "Jelly_Black", "Jelly_Milk", "Jelly_White", "Noblesse", "Noir_authentique", "Passion_au_lait", "Stracciatella", "Tentation_noir", "Triangolo"]

def get_yolo_to_col():
    return {
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


def print_total_paramters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")


############### Module Training and Evaluation ###############
def train_model(model: torch.nn.Module,EPOCHS: int, optimizer: torch.optim.Optimizer, device: torch.device, per_epoch_save: int, loader: torch.utils.data.DataLoader, name_of_saved_pt : str ):
    epoch_prog_bar = tqdm(range(EPOCHS), desc="Training...", leave=False, position=0, total=EPOCHS)
    BEST_LOSS = np.inf
    TOTAL_BATCHES = len(loader)
    loop_prog_bar = tqdm(loader, desc="Processing Image Batch...", leave=False, position=1, total=TOTAL_BATCHES)
    for epoch in epoch_prog_bar:
        epoch_prog_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        
        for idx, (imgs, targets) in enumerate(loop_prog_bar):
            loop_prog_bar.set_description(f"Processing Image Batch {idx+1}/{TOTAL_BATCHES}")
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = yolo_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            loop_prog_bar.set_postfix(loss=loss.item())
    
        if (epoch + 1) % per_epoch_save == 0 or (epoch + 1) == EPOCHS:
            model_path = "models/" + f"{name_of_saved_pt}_{epoch+1}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
    
        if total_loss < BEST_LOSS:
            BEST_LOSS = total_loss
            best_model_path = "models/" + f"{name_of_saved_pt}_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best loss: {BEST_LOSS:.4f}")
            print(f"Best model saved to: {best_model_path}")

        print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {total_loss:.4f}")

    return model


def run_inference(model: torch.nn.Module, path_to_image: Union[List[str], str], IMG_PARAM: dict, conf_threshold: float = 0.2):
    
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    # === Image Transform ===
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if type(path_to_image) == str:
        images_path = [path_to_image]

    for img_path in images_path:
        img_orig = Image.open(img_path).convert("RGB")

        img_resized = img_orig.resize((IMG_SIZE, IMG_SIZE))
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = output.permute(0, 1, 3, 4, 2)
            preds = output[0].cpu().numpy()
        
        yield preds

#without NMS
def use_model_on_images_without_nms(root_dir: str, number_of_images: int, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2):
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    NUM_CLASSES = IMG_PARAM["NUM_CLASSES"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    CLASS_NAMES = get_class_names()
    # Get test images
    test_dir = Path(f"{root_dir}/test/images")
    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]
    
    sampled_paths = np.random.choice(image_paths, size=number_of_images, replace=False).tolist()

    for idx, preds in enumerate(run_inference(model, sampled_paths, IMG_PARAM, conf_threshold)):
        img_resized = cv2.imread(sampled_paths[idx])
        img_resized = cv2.resize(img_resized, (IMG_SIZE, IMG_SIZE))
        img_path = sampled_paths[idx]
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

        for box in boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)
        
            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_resized, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_resized)
        plt.title(f"Prediction (Resized): {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()


# with NMS
def use_model_on_images_with_nms(root_dir: str, number_of_images: int, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.13, min_dist: float = 40.0):
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    NUM_CLASSES = IMG_PARAM["NUM_CLASSES"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    CLASS_NAMES = get_class_names()

    test_dir = Path(f"{root_dir}/test/images")
    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]
    
    sampled_paths = np.random.choice(image_paths, size=number_of_images, replace=False).tolist()
 
    for idx, preds in run_inference(model, sampled_paths, IMG_PARAM, conf_threshold):
        img_resized = cv2.imread(sampled_paths[idx])
        img_resized = cv2.resize(img_resized, (IMG_SIZE, IMG_SIZE))
        img_path = sampled_paths[idx]

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

        for box in kept_boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)

            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_resized, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show result
        plt.figure(figsize=(8, 8))
        plt.imshow(img_resized)
        plt.title(f"Prediction (Filtered): {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

def find_image_path_by_partial_name(test_dir, partial_name):
    for f in os.listdir(test_dir):
        if partial_name.split(".")[0] in f and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(test_dir, f)
    return None

def use_model_on_image_without_nms(root_dir: str, image_names: list, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2):
    # Image and model parameters
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    CLASS_NAMES = get_class_names()

    test_dir = Path(f"{root_dir}/test/images")
    found_partial_names = [find_image_path_by_partial_name(test_dir, name) for name in image_names] 
    img_paths = list(filter(None, found_partial_names))

    for idx, preds in run_inference(model, img_paths, IMG_PARAM, conf_threshold):
        img_resized = cv2.imread(img_paths[idx])
        img_resized = cv2.resize(img_resized, (IMG_SIZE, IMG_SIZE))
        img_path = img_paths[idx]
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

        for box in boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)
        
            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_resized, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_resized)
        plt.title(f"Prediction (Resized): {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()


def use_model_on_image_with_nms(root_dir: str, image_names: list, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2, min_dist: float = 40.0):
    # Image and model parameters
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    CLASS_NAMES = get_class_names()

    test_dir = Path(f"{root_dir}/test/images")
    found_partial_names = [find_image_path_by_partial_name(test_dir, name) for name in image_names] 
    img_paths = list(filter(None, found_partial_names))

    for idx, preds in run_inference(model, img_paths, IMG_PARAM, conf_threshold):
        img_resized = cv2.imread(img_paths[idx])
        img_resized = cv2.resize(img_resized, (IMG_SIZE, IMG_SIZE))
        img_path = img_paths[idx]

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

        for box in kept_boxes:
            x, y, bw, bh, conf, cls_id = box
            x1 = int((x - bw / 2) * IMG_SIZE)
            y1 = int((y - bh / 2) * IMG_SIZE)
            x2 = int((x + bw / 2) * IMG_SIZE)
            y2 = int((y + bh / 2) * IMG_SIZE)

            label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_resized, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show result
        plt.figure(figsize=(8, 8))
        plt.imshow(img_resized)
        plt.title(f"Prediction (Filtered): {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

##############################################################

############### Submission Exporting Functions ###############
def CREATE_TABLE_FROM_PT(data_root_dir: str, output_dir_root: str, model_path: str, model_class: torch.nn.Module, IMG_PARAM, conf_threshold: float = 0.2):
    
    COLS = list(get_yolo_to_col().values())
    YOLO_NAMES = get_yolo_names()
    YOLO_TO_COL = get_yolo_to_col()
    NUM_CLASSES = len(get_class_names())
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]

    test_dir = Path(f"{data_root_dir}/test/images")
    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    filename_pattern_match = re.compile(r'L[0-9]{7}')        

    all_rows = []

    for idx, preds in tqdm(enumerate(run_inference(model, image_paths, IMG_PARAM, conf_threshold)), total=len(image_paths), desc="Processing Images"):
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
        filename = os.path.basename(image_paths[idx])

        re_matches = filename_pattern_match.findall(filename)
        if len(re_matches) == 0:
            print(f"No ID found in filename: {filename}")
            continue
        elif len(re_matches) > 1:
            print(f"Multiple IDs found in filename: {filename}")
            continue

        file_id = re_matches[0]
        
        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)


    df = pd.DataFrame(all_rows)
    df = df.set_index("id").sort_index().reset_index()
    df[COLS] = df[COLS].astype(int)
    
    check_df(df, df_name="Generated Submission")
    output_path = Path(f"{output_dir_root}/submission_conf{int(conf_threshold * 100)}.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved formatted CSV to: {output_path}")
    return df


def CREATE_TABLE_FROM_PT_NMS(data_root_dir: str, output_dir_root: str, model_path: str, model_class: torch.nn.Module,IMG_PARAM, conf_threshold: float = 0.2, min_dist: float = 30.0):
    
    COLS = list(get_yolo_to_col().values())
    YOLO_NAMES = get_yolo_names()
    YOLO_TO_COL = get_yolo_to_col()
    NUM_CLASSES = len(get_class_names())
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]


    test_dir = Path(f"{data_root_dir}/test/images")
    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    filename_pattern_match = re.compile(r'L[0-9]{7}')        

    all_rows = []

    for idx, preds in tqdm(enumerate(run_inference(model, image_paths, IMG_PARAM, conf_threshold)), total=len(image_paths), desc="Processing Images"):

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
        filename = os.path.basename(image_paths[idx])

        re_matches = filename_pattern_match.findall(filename)
        if len(re_matches) == 0:
            print(f"No ID found in filename: {filename}")
            continue
        elif len(re_matches) > 1:
            print(f"Multiple IDs found in filename: {filename}")
            continue

        file_id = re_matches[0]

        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)


    df = pd.DataFrame(all_rows)
    df = df.set_index("id").sort_index().reset_index()
    df[COLS] = df[COLS].astype(int)

    # Save file
    check_df(df, df_name="Generated Submission")
    output_path = f"submission_conf{int(conf_threshold * 100)}_dist{int(min_dist)}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved filtered CSV to: {output_path}")
    return df



def create_controller_csv(label_dir='./data/test/labels', output_path='./DoNotSubmitThis_Controller/controller.csv'):
    YOLO_TO_COL = get_yolo_to_col()
    CLASS_NAMES = get_class_names()
    COLS = ['id'] + list(YOLO_TO_COL.values())
    all_rows = []

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        try:
            # Extract numeric ID from file name
            file_id = int(fname.split('_')[0].lstrip('L'))
        except ValueError:
            print(f"Could not find numeric ID in filename: {fname}")
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
    print(f"Saved controller CSV to: {output_path}")

    check_df(df, df_name="Generated Submission")
    return df

def evaluate_f1_score(pred_df: pd.DataFrame, controller_path: str = "./DoNotSubmitThis_Controller/controller.csv") -> float:
    # Load ground truth
    gt_df = pd.read_csv(controller_path)

    # Align both DataFrames by 'id'
    df = pd.merge(gt_df, pred_df, on='id', suffixes=('_true', '_pred'))
    df = df.sort_values('id').reset_index(drop=True)

    class_names = [col for col in gt_df.columns if col != "id"]

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
    print(f"\nMean F1 Score: {mean_f1:.4f}")
    return mean_f1
