import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import time
from tqdm import tqdm

# Assuming DDETR.py and its dependencies (like src.modules) are accessible
from src.models.DDETR import build # Main build function from your DDETR.py
from src.modules.NestedTensor import nested_tensor_from_tensor_list

# --- Configuration & Arguments ---
class Args:
    def __init__(self):
        # Model parameters - ADJUST THESE TO CONTROL MODEL SIZE AND PERFORMANCE
        self.hidden_dim = 128  # Transformer d_model
        self.nheads = 8        # Number of attention heads
        self.enc_layers = 3    # Number of encoder layers
        self.dec_layers = 3    # Number of decoder layers
        self.dim_feedforward = 1024 # Dimension of FFNs in transformer
        self.num_queries = 100 # Number of object queries
        self.num_feature_levels = 1 # Crucial: Set to 1 for your simplified model
        self.dropout = 0.1
        self.position_embedding = 'sine' # or 'v2'
        self.lr_backbone = 1e-4
        # Loss coefficients
        self.cls_loss_coef = 2.0
        self.enc_n_points = 4
        self.dec_n_points = 4
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        self.focal_alpha = 0.25

        # Matcher costs
        self.set_cost_class = 2.0
        self.set_cost_bbox = 5.0
        self.set_cost_giou = 2.0

        # Training parameters
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.epochs = 50
        self.batch_size = 2#4 # Adjust based on your GPU memory
        self.num_workers = 2 # For DataLoader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42

        # Data paths (MODIFY THESE)
        #self.train_img_dir = "../data/train/images"
        self.train_img_dir = "/Users/lunnet_1/Downloads/All_Data_Final/train_annotated_augmented/all_augmented_images"
        self.train_ann_dir = "/Users/lunnet_1/Downloads/All_Data_Final/train_annotated_augmented/all_augmented_labels"
        #self.train_ann_dir = "../data/train/labels"
        self.val_img_dir = "../data/test/images"
        self.val_ann_dir = "../data/test/labels"
        self.img_size = 640
        self.num_classes = 13

# --- YOLOv8 Dataset ---
def load_yolo_annotations(ann_path):
    boxes = []
    labels = []
    if not os.path.exists(ann_path):
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            # YOLO format is normalized cx, cy, w, h
            # DDETR expects normalized cx, cy, w, h as well
            boxes.append([cx, cy, w, h])
            labels.append(class_id)
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

class YOLODataset(Dataset):
    def __init__(self, img_dir, ann_dir, img_size, num_classes, transform=None, is_train=True):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        self.is_train = is_train

        self.img_files = os.listdir(img_dir) # Add other extensions if needed

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        #img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        original_w, original_h = image.size

        ann_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        ann_path = os.path.join(self.ann_dir, ann_filename)

        boxes, labels = load_yolo_annotations(ann_path)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        target["orig_size"] = torch.as_tensor([int(original_h), int(original_w)])
        target["size"] = torch.as_tensor([self.img_size, self.img_size]) # After resize

        if self.transform:
            image, target = self.transform(image, target) # Custom transform for DDETR

        return image, target


class DDETRTransform:
    def __init__(self, img_size, is_train):
        self.img_size = img_size
        self.is_train = is_train

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transforms = T.Compose([
            T.Resize((img_size, img_size)), # Resize to fixed size
            normalize,
        ])

    def __call__(self, image, target):
        w, h = image.size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        img_transformed = self.transforms(image)
        target["size"] = torch.as_tensor([self.img_size, self.img_size])

        return img_transformed, target


def collate_fn(batch):
    # images are already tensors
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images_nt = nested_tensor_from_tensor_list(images)
    return images_nt, targets

# --- Training and Evaluation Functions ---
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, print_freq=10):
    model.train()
    criterion.train()
    metric_logger = {"loss": [], "class_error": []} # Simplified logger
    start_time = time.time()

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = {k: v for k, v in loss_dict.items()}
        loss_value = losses.item()

        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            # Consider saving model or other diagnostics here
            # sys.exit(1) # Or raise an error

        optimizer.zero_grad()
        losses.backward()
        # Optional: Gradient clipping if needed
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        metric_logger["loss"].append(loss_value)
        if 'class_error' in loss_dict_reduced: # From loss_labels
             metric_logger["class_error"].append(loss_dict_reduced['class_error'].item())


        if (i + 1) % print_freq == 0:
            elapsed_time = time.time() - start_time
            avg_loss = np.mean(metric_logger["loss"][-print_freq:])
            avg_class_err = np.mean(metric_logger["class_error"][-print_freq:]) if metric_logger["class_error"] else -1.0
            print(f"Epoch: [{epoch+1}] Batch: [{i+1}/{len(data_loader)}] "
                  f"Loss: {avg_loss:.4f} ClassError: {avg_class_err:.2f}% "
                  f"Time: {elapsed_time:.2f}s")
            start_time = time.time() # Reset for next print interval

    return np.mean(metric_logger["loss"]), np.mean(metric_logger["class_error"]) if metric_logger["class_error"] else -1.0

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    metric_logger = {"loss": [], "class_error": []}

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) # Not strictly needed for eval usually

        metric_logger["loss"].append(loss_dict['loss_ce'].item()) # Example: just track one loss component
        if 'class_error' in loss_dict:
             metric_logger["class_error"].append(loss_dict['class_error'].item())

    avg_loss = np.mean(metric_logger["loss"])
    avg_class_err = np.mean(metric_logger["class_error"]) if metric_logger["class_error"] else -1.0
    print(f"Validation: Avg Loss: {avg_loss:.4f} Avg ClassError: {avg_class_err:.2f}%")
    return avg_loss, avg_class_err

# --- Main Execution ---
if __name__ == "__main__":
    args = Args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Sanity check paths ---
    if not os.path.isdir(args.train_img_dir) or not os.path.isdir(args.train_ann_dir):
        print("ERROR: Training image or annotation directory not found!")
        print("Please set 'train_img_dir' and 'train_ann_dir' in Args().")
        exit()
    if not os.path.isdir(args.val_img_dir) or not os.path.isdir(args.val_ann_dir):
        print("ERROR: Validation image or annotation directory not found!")
        print("Please set 'val_img_dir' and 'val_ann_dir' in Args().")
        exit()


    # --- Build Model, Criterion, Postprocessor ---
    # Ensure num_classes in DDETR.py's build function matches args.num_classes
    # For this example, we assume DDETR.py's build() uses a fixed num_classes (e.g., 13)
    # or you modify it to take num_classes from args.
    # If DDETR.py build() takes args directly, ensure it gets args.num_classes
    print(f"Building model with {args.num_classes} classes (ensure DDETR.py uses this).")

    model, criterion, postprocessors = build(args)
    model.to(args.device)
    criterion.to(args.device) # Criterion also needs to be on device if it has parameters

    # --- Parameter Count Check ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable model parameters: {trainable_params:,}")

    if total_params > 12_000_000:
        print(f"WARNING: Model parameter count ({total_params:,}) exceeds 12M.")
        print("Consider reducing hidden_dim, enc_layers, dec_layers, or dim_feedforward in Args.")
    else:
        print(f"Model parameter count ({total_params:,}) is within the 12M limit.")


    # --- Optimizer ---
    # Standard DETR optimizer setup
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone, # This is very low for the simplified backbone, likely no effect
        },
    ]

    # Since your backbone is very simple and part of the main model, you can simplify:
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_dicts[0]['params'], lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # Adjust step_size

    # --- Datasets and DataLoaders ---
    transform_train = DDETRTransform(img_size=args.img_size, is_train=True)
    transform_val = DDETRTransform(img_size=args.img_size, is_train=False)

    
    dataset_train = YOLODataset(args.train_img_dir, args.train_ann_dir, args.img_size, args.num_classes, transform=transform_train)
    dataset_val = YOLODataset(args.val_img_dir, args.val_ann_dir, args.img_size, args.num_classes, transform=transform_val)

    
    if not dataset_train:
        print("No training images found. Check 'train_img_dir' and image extensions.")
        exit()
    if not dataset_val:
        print("No validation images found. Check 'val_img_dir' and image extensions.")
        exit()

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"Using device: {args.device}")
    print(f"Found {len(dataset_train)} training images and {len(dataset_val)} validation images.")

    # Track history
    history = {
        "train_loss": [],
        "train_class_err": [],
        "val_loss": [],
        "val_class_err": [],
    }

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in tqdm(range(args.epochs), desc="Epoch Loop"):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss, train_class_err = train_one_epoch(model, criterion, data_loader_train, optimizer, args.device, epoch)
        lr_scheduler.step()

        val_loss, val_class_err = evaluate(model, criterion, data_loader_val, args.device)
        
        history["train_loss"].append(train_loss)
        history["train_class_err"].append(train_class_err)
        history["val_loss"].append(val_loss)
        history["val_class_err"].append(val_class_err)

        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Train ClassErr: {train_class_err:.2f}%")
        print(f"Epoch {epoch+1} Summary: Val Loss: {val_loss:.4f},   Val ClassErr: {val_class_err:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ddetr_simplified_best.pth")
            print("Saved best model checkpoint.")

    print("\nTraining finished.")
    # Optional: Load best model and run final evaluation
    # model.load_state_dict(torch.load("ddetr_simplified_best.pth"))
    # final_val_loss, final_val_class_err = evaluate(model, criterion, data_loader_val, args.device)
    # print(f"Final Validation on Best Model: Loss: {final_val_loss:.4f}, ClassError: {final_val_class_err:.2f}%")

    # --- Plotting ---
    epochs = list(range(1, args.epochs + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker='o')
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot Class Error
    ax2.plot(epochs, history["train_class_err"], label="Train Class Error", marker='s')
    ax2.plot(epochs, history["val_class_err"], label="Val Class Error", marker='s')
    ax2.set_title("Class Error over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("loss_and_error_plots.png")
    plt.show()


    # --- Save history to JSON ---
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("Saved plots and training history to disk.")