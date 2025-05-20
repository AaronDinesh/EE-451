from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image, ImageFilter
from PIL import ImageDraw, ImageEnhance
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm
import numpy as np
from src.lossfunctions.YOLOLoss import yolo_loss
import cv2
from typing import Union, List
import re
import pandas as pd
from check import check_df, COLS, IDS, TermFormat
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
import time


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


##########

def test_map50_on_training_set(model, loader, device, iou_threshold = 0.000001):
    model.eval()  # Set to evaluation mode
    all_pred_boxes = []
    all_true_boxes = []

    progress_bar = tqdm(total=len(loader), desc="Processing Training Set", position=0, leave=True, ncols=100, colour='green')

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)

            # Extract predictions
            batch_size, anchors, _, grid_h, grid_w = outputs.shape
            obj_scores = torch.sigmoid(outputs[:, :, 4, :, :])
            mask = obj_scores > 0.5

            # Permute outputs for easier indexing
            outputs = outputs.permute(0, 1, 3, 4, 2)

            # Get the positions of valid predictions
            batch_idx, anchor_idx, gy, gx = torch.nonzero(mask, as_tuple=True)

            # Extract bounding boxes
            pred_boxes = outputs[mask, :4]
            x = (gx + torch.sigmoid(pred_boxes[..., 0])) / grid_w
            y = (gy + torch.sigmoid(pred_boxes[..., 1])) / grid_h
            bw = pred_boxes[..., 2].pow(2)
            bh = pred_boxes[..., 3].pow(2)

            # Concatenate into final format
            batch_pred_boxes = torch.stack([x, y, bw, bh, obj_scores[mask]], dim=1)
            for idx in range(batch_size):
                all_pred_boxes.extend(batch_pred_boxes[batch_idx == idx].cpu().tolist())

            # Extract ground truth boxes
            objectness_mask = targets[..., 4] > 0.5
            boxes = targets[..., :4][objectness_mask]
            all_true_boxes.extend(boxes.cpu().tolist())

            # 🔄 Update the progress bar
            progress_bar.update(1)
    # ✅ Close the progress bar after completion
    progress_bar.close()
    # Run mAP50 calculation
    precision, recall = calculate_map50(all_pred_boxes, all_true_boxes, iou_threshold)
    mAP50_score = (precision + recall) / 2
    return mAP50_score

##########

############### Module Training and Evaluation ###############

def create_table_from_pt_nms_for_training_F1(model,IMG_PARAM, conf_threshold: float = 0.5, min_dist: float = 40.0):
    
    #TODO: Check and throw error if some of the required parameters are None

    COLS = list(get_yolo_to_col().values())
    YOLO_NAMES = get_yolo_names()
    YOLO_TO_COL = get_yolo_to_col()
    NUM_CLASSES = len(get_class_names())
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]

    current_dir = os.getcwd()
    test_dir = Path(f"{current_dir}/data/test/images")

    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]

    
    filename_pattern_match = re.compile(r'L[0-9]{7}')        

    all_rows = []

    for idx, preds in tqdm(enumerate(run_inference(model, image_paths, IMG_PARAM, conf_threshold)), 
                       total=len(image_paths), 
                       desc="Processing Images", position=0, leave=True):

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

        file_id = int(re_matches[0][1:])

        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)


    df = pd.DataFrame(all_rows)
    df = df.set_index("id").sort_index().reset_index()
    df[COLS] = df[COLS].astype(int)

    # Save file
    check_df(df, df_name="Generated Submission")
    return df


def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is defined as [x_center, y_center, width, height].
    """
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def calculate_map50(pred_boxes, true_boxes, iou_threshold= 0.000001):
    """
    Calculate mAP@0.5 for a batch of images.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    detected_boxes = []

    # Add a progress bar to the outer loop
    print("🔄 Calculating mAP50 for all predictions...")
    for pred in tqdm(sorted(pred_boxes, key=lambda x: x[4], reverse=True), desc="Processing Predictions"):
        matched = False
        
        # Add a nested progress bar for each GT check
        for i, gt in enumerate(true_boxes):
            if i not in detected_boxes and iou(pred, gt) > iou_threshold:
                true_positives += 1
                detected_boxes.append(i)
                matched = True
                break
        
        if not matched:
            false_positives += 1

    false_negatives = len(true_boxes) - len(detected_boxes)
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    return precision, recall


def train_model(model: torch.nn.Module, total_epochs: int, optimizer: torch.optim.Optimizer, device: torch.device, 
                per_epoch_save: int, train_loader, test_loader, plotting_callback: callable, 
                name_of_saved_pt: str, pt_save_path: str,IMG_PARAM: dict):
    """
    Train the YOLO model and evaluate mAP50 and F1 on training and test sets.
    """
    training_metrics = {
        "train_loss": [],
        "test_loss": [],
        "mAP50_training": [],
        "mAP50_testing": [],
        "testing-F1": [],
        "epoch": []
    }
    
    best_train_loss = np.inf
    best_f1_score = -np.inf

    for epoch in tqdm(range(total_epochs), desc="Training..."):
        training_metrics["epoch"].append(epoch)
        model.train()
        total_train_loss = 0
        all_pred_boxes = []
        all_true_boxes = []
        f1_scores_train = []

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = yolo_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            with torch.no_grad():
                batch_size, anchors, _, grid_h, grid_w = outputs.shape
                
                # Reshape and apply sigmoid to objectness scores
                obj_scores = torch.sigmoid(outputs[:, :, 4, :, :]).permute(0, 1, 2, 3)
                mask = obj_scores > 0.5

                # Get coordinates (x, y, w, h) for the predictions
                outputs = outputs.permute(0, 1, 3, 4, 2)
                pred_boxes = outputs[mask, :4]
                
                # Get the positions of the valid predictions
                batch_idx, anchor_idx, gy, gx = torch.nonzero(mask, as_tuple=True)
                x = (gx + torch.sigmoid(pred_boxes[..., 0])) / grid_w
                y = (gy + torch.sigmoid(pred_boxes[..., 1])) / grid_h
                bw = pred_boxes[..., 2].pow(2)
                bh = pred_boxes[..., 3].pow(2)
                
                # Extract class probabilities
                class_probs = torch.softmax(outputs[..., 5:], dim=-1)
                valid_class_probs = class_probs[mask]
                cls_ids = torch.argmax(valid_class_probs, dim=-1)

                # Stack all predictions
                batch_pred_boxes = torch.stack([x, y, bw, bh, obj_scores[mask], cls_ids.float()], dim=1)

                # Append to all predictions
                for idx in range(batch_size):
                    all_pred_boxes.extend(batch_pred_boxes[batch_idx == idx].cpu().tolist())

        # ➡️ Calculate mAP50 for the **Training Set**
        if(epoch > 5):
            mAP50_score_training = test_map50_on_training_set(model,train_loader,device)
        else:
            mAP50_score_training = 0
        training_metrics["mAP50_training"].append(mAP50_score_training)

        # ➡️ Append train loss
        avg_train_loss = total_train_loss / len(train_loader)
        training_metrics["train_loss"].append(avg_train_loss)

        

        # --- Testing Loop ---
        model.eval()
        total_test_loss = 0
        f1_scores_test = []

        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = yolo_loss(outputs, targets)
                total_test_loss += loss.item()

        # ➡️ Calculate mAP50 for the **Testing Set**
        if(epoch > 5):
            mAP50_score_testing = test_map50_on_training_set(model,test_loader,device)
        else:
            mAP50_score_testing = 0
        
        training_metrics["mAP50_testing"].append(mAP50_score_testing)

        # ➡️ Calculate average F1 for test set
        df = create_table_from_pt_nms_for_training_F1(model,IMG_PARAM)
        f1_test = evaluate_f1_score_test(df)
        training_metrics["testing-F1"].append(f1_test)

        # ➡️ Append test loss
        avg_test_loss = total_test_loss / len(test_loader)
        training_metrics["test_loss"].append(avg_test_loss)

        # ➡️ Logging
        print(f"Epoch {epoch+1}/{total_epochs} | Mean F1 Test: {f1_test:.4f} | mAP50 Test: {mAP50_score_testing:.4f} | Test Loss: {total_test_loss:.4f}")
        # ➡️ Logging
        print(f"Epoch {epoch+1}/{total_epochs} | mAP50 Train: {mAP50_score_training:.4f} | Train Loss: {total_train_loss:.4f}")
        
        # ➡️ Save the model
        if (epoch + 1) % per_epoch_save == 0 or (epoch + 1) == total_epochs:
            model_path = os.path.join(pt_save_path, f"{name_of_saved_pt}_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to: {model_path}")
    
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss
            best_model_path = os.path.join(pt_save_path, f"{name_of_saved_pt}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best loss: {best_train_loss:.4f}")
            print(f"✅ Best model saved to: {best_model_path}")

        if epoch > 80 and f1_test > best_f1_score:
            best_f1_score = f1_test
            f1_best_model = type(model)()  # Create a new instance of the model
            f1_best_model.load_state_dict(model.state_dict())
            f1_best_model.to(device)
            best_model_path = os.path.join(pt_save_path, f"{name_of_saved_pt}_BEST_F1_SCORE_MODEL.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best f1: {best_f1_score:.4f}")
            print(f"✅ Best f1 scored model saved to: {best_model_path}")

    
    print(f"BEST F1 SCORE FROM TRAINING: {best_f1_score}")
    return model, training_metrics, f1_best_model



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

    # 📝 If it's a string, convert it to a single-item list
    if isinstance(path_to_image, str):
        images_path = [path_to_image]
        print("Input is a single string, converting to list.")
    elif isinstance(path_to_image, list):
        # Convert Path objects to strings
        images_path = [str(p) if isinstance(p, Path) else p for p in path_to_image]
    else:
        raise ValueError(f"Expected a string or list of strings, but got {type(path_to_image)}")

    
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
        img_resized = cv2.imread(str(sampled_paths[idx]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
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

    test_dir = Path(f"{root_dir}/images")
    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]
    
    sampled_paths = np.random.choice(image_paths, size=number_of_images, replace=False).tolist()
 
    for idx, preds in enumerate(run_inference(model, sampled_paths, IMG_PARAM, conf_threshold)):
        img_resized = cv2.imread(str(sampled_paths[idx]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
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

def use_model_on_image_without_nms_by_name(root_dir: str, image_names: list, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2):
    # Image and model parameters
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    CLASS_NAMES = get_class_names()

    test_dir = Path(f"{root_dir}/images")
    found_partial_names = [find_image_path_by_partial_name(test_dir, name) for name in image_names] 
    img_paths = list(filter(None, found_partial_names))

    for idx, preds in enumerate(run_inference(model, img_paths, IMG_PARAM, conf_threshold)):
        img_resized = cv2.imread(str(img_paths[idx]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
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


def use_model_on_image_with_nms_by_name(root_dir: str, image_names: list, model: torch.nn.Module, IMG_PARAM: dict, conf_threshold: float = 0.2, min_dist: float = 40.0):
    # Image and model parameters
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]
    CLASS_NAMES = get_class_names()

    test_dir = Path(f"{root_dir}/images")
    found_partial_names = [find_image_path_by_partial_name(test_dir, name) for name in image_names] 
    img_paths = list(filter(None, found_partial_names))

    for idx, preds in enumerate(run_inference(model, img_paths, IMG_PARAM, conf_threshold)):
        img_resized = cv2.imread(str(img_paths[idx]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  
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
        fig = plt.gcf()
        
        return fig
        

##############################################################

############### Submission Exporting Functions ###############
def create_table_from_pt_without_nms(data_root_dir: str, output_dir_root: str, model_path: str, model_class: torch.nn.Module, IMG_PARAM, conf_threshold: float = 0.2):
    
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

    for idx, preds in enumerate(tqdm(run_inference(model, image_paths, IMG_PARAM, conf_threshold), total=len(image_paths), desc="Processing Images")):
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

        file_id = int(re_matches[0][1:])
        
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


def create_table_from_pt_nms(data_root_dir: str=None, output_dir_root: str=None, model_path: str=None, model_class: torch.nn.Module=None,IMG_PARAM=None, conf_threshold: float = 0.2, min_dist: float = 30.0):
    
    #TODO: Check and throw error if some of the required parameters are None

    COLS = list(get_yolo_to_col().values())
    YOLO_NAMES = get_yolo_names()
    YOLO_TO_COL = get_yolo_to_col()
    NUM_CLASSES = len(get_class_names())
    IMG_SIZE = IMG_PARAM["IMG_SIZE"]
    ANCHORS = IMG_PARAM["ANCHOR"]
    GRID_SIZE = IMG_PARAM["GRID_SIZE"]


    test_dir = Path(f"{data_root_dir}")
    img_exts = [".jpg", ".png", ".jpeg"]
    image_paths = [f for f in test_dir.iterdir() if f.suffix.lower() in img_exts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    filename_pattern_match = re.compile(r'L[0-9]{7}')        

    all_rows = []

    for idx, preds in enumerate(tqdm(run_inference(model, image_paths, IMG_PARAM, conf_threshold), total=len(image_paths), desc="Processing Images")):

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

        file_id = int(re_matches[0][1:])

        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)


    df = pd.DataFrame(all_rows)
    df = df.set_index("id").sort_index().reset_index()
    df[COLS] = df[COLS].astype(int)

    # Save file
    check_df(df, df_name="Generated Submission")
    if output_dir_root is None:
        output_path = f"BEST_SUBMISSION.csv"
        print(f"Saving to the default location {os.getcwd()}/{output_path}")
    else:
        output_path = Path(f'{output_dir_root}/BEST_SUBMISSION.csv')
        print(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"\nSaved filtered CSV to: {output_path}")
    return df



def create_controller_csv_test(label_dir='./data/test/labels', output_path='./DoNotSubmitThis_Controller/controller_test.csv'):
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

def create_controller_csv_train(label_dir='./data/train/labels', image_dir='./data/train/images', output_path='./DoNotSubmitThis_Controller/controller_train.csv'):
    YOLO_TO_COL = get_yolo_to_col()
    CLASS_NAMES = get_class_names()
    COLS = ['id'] + list(YOLO_TO_COL.values())
    all_rows = []

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        # Find the corresponding image file
        image_name = fname.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"No corresponding image found for label: {fname}")
            continue
        
        # Use the full image name as the ID (without extension)
        file_id = image_name

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
    return df

def evaluate_f1_score_test(pred_df: pd.DataFrame, controller_path: str = "./DoNotSubmitThis_Controller/controller_test.csv") -> float:
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
    # print(f"\nMean F1 Score: {mean_f1:.4f}")
    return mean_f1

def evaluate_f1_score_train(pred_df: pd.DataFrame, controller_path: str = "./DoNotSubmitThis_Controller/controller_train.csv") -> float:
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
