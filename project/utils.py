from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import os



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


# -------------------------------
# Data Augmentation 
# -------------------------------

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

class RandomGaussianBlur:
    def __init__(self, radius_min=0.1, radius_max=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))

class RandomShadow:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        shadow = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(shadow)
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, img.width // 2)
            y1 = random.randint(0, img.height)
            x2 = random.randint(img.width // 2, img.width)
            y2 = random.randint(0, img.height)
            x0, x1_sorted = sorted([x1, x2])
            y0, y1_sorted = sorted([y1, y2])
            draw.ellipse([x0, y0, x1_sorted, y1_sorted], fill=random.randint(60, 120))

        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=random.uniform(10, 30)))
        shadow = shadow.convert("RGB")
        shadow = ImageEnhance.Brightness(shadow).enhance(0.6)
        img = Image.blend(img, shadow, alpha=0.5)
        return img

class RandomBlackLines:
    def __init__(self, p=0.5, max_lines=5, max_thickness=5):
        self.p = p
        self.max_lines = max_lines
        self.max_thickness = max_thickness

    def __call__(self, img):
        if random.random() > self.p:
            return img

        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(1, self.max_lines)):
            x1, y1 = random.randint(0, img.width), random.randint(0, img.height)
            x2, y2 = random.randint(0, img.width), random.randint(0, img.height)
            thickness = random.randint(2, self.max_thickness)
            draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=thickness)

        return img

# -------------------------------
# Transform 
# -------------------------------


class RandomSubsetTransform:
    def __init__(self, transforms, num_choices=2):
        self.transforms = transforms
        self.num_choices = num_choices

    def __call__(self, img):
        for t in random.sample(self.transforms, self.num_choices):
            img = t(img)
        return img
    

def get_train_transform():
    random_augmentations = [
        T.RandomRotation(30),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
        RandomGaussianBlur(),
        RandomShadow(p=0.7),
        RandomBlackLines(p=0.5, max_lines=5, max_thickness=4),
    ]

    return T.Compose([
        T.RandomResizedCrop(150),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        RandomSubsetTransform(random_augmentations, num_choices=2),
        T.ToTensor(),
        RandomGaussianNoise(0., 0.02, p=0.3),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

# -------------------------------
#  CutPaste Chocolate Dataset
# -------------------------------

class ChocolateCutPasteDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, cutpaste_p=0.5):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() == ".jpg"])
        self.transform = transform
        self.cutpaste_p = cutpaste_p

    def __len__(self):
        return len(self.image_files)

    def load_labels(self, label_path, img_width, img_height):
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                x1 = int((x - w / 2) * img_width)
                y1 = int((y - h / 2) * img_height)
                x2 = int((x + w / 2) * img_width)
                y2 = int((y + h / 2) * img_height)
                boxes.append((int(cls), x1, y1, x2, y2))
        return boxes

    def __getitem__(self, idx):
        img_name = self.image_files[idx].name
        img_path = self.image_dir / img_name
        label_path = self.label_dir / img_name.replace(".jpg", ".txt")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        boxes = self.load_labels(label_path, width, height)

        # CutPaste logic
        if random.random() < self.cutpaste_p and len(boxes) > 0:
            donor_idx = random.randint(0, len(self.image_files) - 1)
            donor_name = self.image_files[donor_idx].name
            donor_img = Image.open(self.image_dir / donor_name).convert("RGB")
            donor_boxes = self.load_labels(self.label_dir / donor_name.replace(".jpg", ".txt"), *donor_img.size)

            if donor_boxes:
                cls, x1, y1, x2, y2 = random.choice(donor_boxes)
                donor_crop = donor_img.crop((x1, y1, x2, y2))

                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < width and crop_h < height:
                    paste_x = random.randint(0, width - crop_w)
                    paste_y = random.randint(0, height - crop_h)
                    image.paste(donor_crop, (paste_x, paste_y))

                    new_box = (cls, paste_x, paste_y, paste_x + crop_w, paste_y + crop_h)
                    boxes.append(new_box)

        if self.transform:
            image = self.transform(image)

        boxes_xyxy = [b[1:] for b in boxes]
        labels = [b[0] for b in boxes]

        target = {
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target