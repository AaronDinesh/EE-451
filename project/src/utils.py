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

        if epoch > -1 and f1_test > best_f1_score:
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


    test_dir = Path(f"{data_root_dir}/test/images")
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

## DATA AUGMENTATION FUNCTIONS

# Removes bounding boxes that are too small compared to image size
def filter_boxes_by_relative_size(boxes, image_width, image_height, min_size_ratio=0.03):
    min_w = image_width * min_size_ratio
    min_h = image_height * min_size_ratio

    filtered = []
    for cls, x1, y1, x2, y2 in boxes:
        w = x2 - x1
        h = y2 - y1
        if w >= min_w and h >= min_h:
            filtered.append((cls, x1, y1, x2, y2))
    return filtered

# Computes IoU (Intersection over Union) between two boxes (to remove pasted images that cover too much of another box)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter_area

    return inter_area / union


class RandomCutPasteWithBoxes:
    def __init__(self, image_dir, label_dir, p=0.8, max_iou=0.3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.p = p
        self.max_iou = max_iou

    def load_labels(self, label_path, img_width, img_height):
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                x1 = (x - w / 2) * img_width
                y1 = (y - h / 2) * img_height
                x2 = (x + w / 2) * img_width
                y2 = (y + h / 2) * img_height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __call__(self, img, target):
        if random.random() > self.p or len(target["boxes"]) == 0:
            return img, target

        width, height = img.size

        donor_idx = random.randint(0, len(self.image_files) - 1)
        donor_name = self.image_files[donor_idx]
        donor_img_path = os.path.join(self.image_dir, donor_name)
        donor_label_path = os.path.join(self.label_dir, donor_name.replace(".jpg", ".txt"))

        donor_img = Image.open(donor_img_path).convert("RGB")
        donor_w, donor_h = donor_img.size

        donor_boxes, donor_labels = self.load_labels(donor_label_path, donor_w, donor_h)
        if len(donor_boxes) == 0:
            return img, target

        idx = random.randint(0, len(donor_boxes) - 1)
        x1, y1, x2, y2 = donor_boxes[idx].tolist()
        cls = donor_labels[idx].item()

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w >= width or crop_h >= height:
            return img, target

        donor_crop = donor_img.crop((x1, y1, x2, y2))

        paste_x = random.randint(0, width - int(crop_w))
        paste_y = random.randint(0, height - int(crop_h))
        new_box = [paste_x, paste_y, paste_x + crop_w, paste_y + crop_h]

        # Avoid pasting over existing boxes
        for box in target["boxes"]:
            if compute_iou(box.tolist(), new_box) > self.max_iou:
                return img, target

        img.paste(donor_crop, (paste_x, paste_y))
        new_boxes = torch.cat([target["boxes"], torch.tensor([new_box], dtype=torch.float32)], dim=0)
        new_labels = torch.cat([target["labels"], torch.tensor([cls], dtype=torch.long)], dim=0)

        target["boxes"] = new_boxes
        target["labels"] = new_labels
        return img, target


class RandomResizedCropWithBoxes:
    def __init__(self, output_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio

    def get_crop_params(self, img):
        width, height = img.size
        area = width * height

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            crop_w = int(round((target_area * aspect_ratio) ** 0.5))
            crop_h = int(round((target_area / aspect_ratio) ** 0.5))

            if crop_w <= width and crop_h <= height:
                top = random.randint(0, height - crop_h)
                left = random.randint(0, width - crop_w)
                return top, left, crop_h, crop_w

        # fallback to center crop
        in_ratio = width / height
        if in_ratio < self.ratio[0]:
            crop_w = width
            crop_h = int(round(width / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_h = height
            crop_w = int(round(height * self.ratio[1]))
        else:
            crop_w, crop_h = width, height
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return top, left, crop_h, crop_w

    def __call__(self, img, target):
        orig_w, orig_h = img.size
        top, left, crop_h, crop_w = self.get_crop_params(img)

        # Crop and resize image to output size (e.g., 416x416)
        img = TF.resized_crop(img, top, left, crop_h, crop_w, size=(self.output_size, self.output_size))

        scale_x = self.output_size / crop_w
        scale_y = self.output_size / crop_h

        boxes = target["boxes"]
        labels = target["labels"]
        boxes_out = []
        labels_out = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            # Shift to crop coords
            x1_new = (x1 - left) * scale_x
            y1_new = (y1 - top) * scale_y
            x2_new = (x2 - left) * scale_x
            y2_new = (y2 - top) * scale_y

            # Clip to image bounds
            x1_new = max(0, min(self.output_size, x1_new))
            y1_new = max(0, min(self.output_size, y1_new))
            x2_new = max(0, min(self.output_size, x2_new))
            y2_new = max(0, min(self.output_size, y2_new))

            # Skip invalid boxes (too small or inverted)
            if (x2_new - x1_new) >= 1 and (y2_new - y1_new) >= 1:
                boxes_out.append([x1_new, y1_new, x2_new, y2_new])
                labels_out.append(labels[i])

        if boxes_out:
            target["boxes"] = torch.tensor(boxes_out, dtype=torch.float32)
            target["labels"] = torch.tensor(labels_out, dtype=torch.long)
        else:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.long)

        return img, target




class RandomVerticalFlipWithBoxes:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        img = TF.vflip(img)
        h = img.size[1]

        boxes = target["boxes"].clone()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]  # flip y1, y2
        target["boxes"] = boxes
        return img, target

class RandomBlackLinesWithTarget:
    def __init__(self, p=0.5, max_lines=5, max_thickness=5):
        self.p = p
        self.max_lines = max_lines
        self.max_thickness = max_thickness

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(1, self.max_lines)):
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = random.randint(0, img.width)
            y2 = random.randint(0, img.height)
            thickness = random.randint(2, self.max_thickness)
            draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=thickness)

        return img, target

class RandomGaussianNoise:
    def __init__(self, mean=0., std=0.01, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"

# Applies random Gaussian blur to an image
class RandomGaussianBlur:
    def __init__(self, radius_min=0.1, radius_max=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))
    
class RandomComposeWithTarget:
    def __init__(self, transforms, min_transforms=2, max_transforms=None, shuffle=True):
        self.transforms = transforms
        self.min_transforms = min_transforms
        self.max_transforms = max_transforms or len(transforms)
        self.shuffle = shuffle

    def __call__(self, img, target):
        transforms = self.transforms[:]
        if self.shuffle:
            random.shuffle(transforms)
        num_to_apply = random.randint(self.min_transforms, self.max_transforms)
        chosen = transforms[:num_to_apply]
        for t in chosen:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlipWithBoxes:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        img = TF.hflip(img)
        w = img.size[0]  # width

        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # flip x1 and x2
        target["boxes"] = boxes
        return img, target


# Adds random black lines to simulate occlusions (like headbands)
class RandomBlackLines:
    def __init__(self, p=0.5, max_lines=3, max_thickness=5):
        self.p = p
        self.max_lines = max_lines
        self.max_thickness = max_thickness

    def __call__(self, img):
        if random.random() > self.p:
            return img

        draw = ImageDraw.Draw(img)
        num_lines = random.randint(1, self.max_lines)

        for _ in range(num_lines):
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = random.randint(0, img.width)
            y2 = random.randint(0, img.height)
            thickness = random.randint(2, self.max_thickness)
            draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=thickness)

        return img
    
class RandomBlackLinesWithTarget:
    def __init__(self, p=0.5, max_lines=3, max_thickness=5):
        self.p = p
        self.max_lines = max_lines
        self.max_thickness = max_thickness

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(1, self.max_lines)):
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = random.randint(0, img.width)
            y2 = random.randint(0, img.height)
            thickness = random.randint(2, self.max_thickness)
            draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=thickness)

        return img, target
    

class RandomSubsetTransformWithTarget:
    def __init__(self, transforms, num_choices=2):
        self.transforms = transforms
        self.num_choices = num_choices

    def __call__(self, img, target):
        chosen_transforms = random.sample(self.transforms, min(self.num_choices, len(self.transforms)))
        for t in chosen_transforms:
            img, target = t(img, target)
        return img, target

class WrapperForImageOnlyTransform:
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, img, target):
        img = self.image_transform(img)
        return img, target
    
