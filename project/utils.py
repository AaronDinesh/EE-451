from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml



class yolo_dataset(Dataset):
    def __init__(self, root_dir:str, target_size=(640, 640), img_exts={'.jpg', '.png', '.jpeg'}):
        self.image_dir = Path(f"{root_dir}/images")
        self.label_dir = Path(f"{root_dir}/labels")
        self.image_size = target_size
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor()
        ])
        self.img_exts = img_exts
        self.images = [f for f in self.image_dir.iterdir() if f.suffix.lower() in self.img_exts]
        self.labels = [f for f in self.label_dir.iterdir() if f.suffix.lower() == '.txt']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            w, h = self.image_size[1], self.image_size[0]
        else:
            w, h = image.size

        # Load labels
        boxes = []
        classes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    #split the lines into the class id, center x, center y, width, and height
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    # Convert YOLO format (center, width/height) to (x1, y1, x2, y2) in pixels
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    classes.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {"boxes": boxes, "labels": classes}
        return image, target

def yolo_dataloader(root_dir, image_size=(640, 640), batch_size=4, shuffle=True, collate_fn=None):
    
    dataset = yolo_dataset(root_dir=root_dir, target_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def visualize_yolo_dataloader_batch(images, targets, class_names=None, max_images=4):
    """
    Visualize a batch of YOLO-style images and bounding boxes.

    Parameters:
        images (list of Tensors): Batch of images from the dataloader.
        targets (list of dicts): Corresponding targets with 'boxes' and 'labels'.
        class_names (list of str, optional): List of class names to map from label IDs.
        max_images (int): Max number of images to show from the batch.
    """
    batch_size = len(images)
    num_images = min(batch_size, max_images)
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        img = images[i]
        boxes = targets[i]['boxes']
        labels = targets[i]['labels']

        # Convert tensor image to PIL
        img_np = T.functional.to_pil_image(img)

        ax = plt.subplot(1, num_images, i + 1)
        ax.imshow(img_np)
        ax.axis("off")

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            if class_names:
                label_text = class_names[label]
            else:
                label_text = str(label.item())

            ax.text(x1, y1 - 4, label_text, color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5, pad=1))

    plt.tight_layout()
    plt.show()


def get_class_names(root_dir):
    with open(root_dir / 'data.yaml', 'r') as f:
        data_dir = yaml.safe_load(f)

    return data_dir['names']

def get_train_path(root_dir):
    with open(root_dir / 'data.yaml', 'r') as f:
        data_dir = yaml.safe_load(f)

    return data_dir['train']

def get_val_path(root_dir):
    with open(root_dir / 'data.yaml', 'r') as f:
        data_dir = yaml.safe_load(f)

    return data_dir['val']

def get_test_path(root_dir):
    with open(root_dir / 'data.yaml', 'r') as f:
        data_dir = yaml.safe_load(f)

    return data_dir['test']
