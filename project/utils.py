from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
# Adds random pixel-wise Gaussian noise to the image tensor
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


# Adds random black lines to simulate occlusions (like headbands)
class RandomBlackLines:
    def __init__(self, p=0.5, max_lines=5, max_thickness=5):
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
    
# Performs a random crop and resize while adjusting bounding boxes
class RandomResizedCropWithBoxes:
    def __init__(self, size, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size  # final output size (150)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img):
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = width / height
        if in_ratio < self.ratio[0]:
            w = width
            h = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            h = height
            w = int(round(h * self.ratio[1]))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, boxes):
        i, j, h, w = self.get_params(img)

        # Crop and resize image
        img_cropped = TF.resized_crop(img, top=i, left=j, height=h, width=w, size=(self.size, self.size))

        # Adjust boxes
        boxes_out = []
        scale_x = self.size / w
        scale_y = self.size / h

        for cls, x1, y1, x2, y2 in boxes:
            # Shift by crop origin
            x1_new = (x1 - j) * scale_x
            y1_new = (y1 - i) * scale_y
            x2_new = (x2 - j) * scale_x
            y2_new = (y2 - i) * scale_y

            # Clip to valid bounds
            x1_new = max(0, min(self.size, x1_new))
            y1_new = max(0, min(self.size, y1_new))
            x2_new = max(0, min(self.size, x2_new))
            y2_new = max(0, min(self.size, y2_new))

            # Discard box if completely outside crop
            if x2_new - x1_new > 1 and y2_new - y1_new > 1:
                boxes_out.append((cls, int(x1_new), int(y1_new), int(x2_new), int(y2_new)))

        boxes_out = filter_boxes_by_relative_size(
            boxes_out,
            image_width=self.size,
            image_height=self.size,
            min_size_ratio=0.03  # adjustable
        )
        return img_cropped, boxes_out
    
# Random horizontal flip with box adjustment
class RandomHorizontalFlipWithBoxes:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes  # no flip

        img = TF.hflip(img)
        w, _ = img.size
        flipped_boxes = []
        for cls, x1, y1, x2, y2 in boxes:
            new_x1 = w - x2
            new_x2 = w - x1
            flipped_boxes.append((cls, new_x1, y1, new_x2, y2))
        return img, flipped_boxes
    
# Random vertical flip with box adjustment
class RandomVerticalFlipWithBoxes:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() > self.p:
            return img, boxes  # no flip

        img = TF.vflip(img)
        _, h = img.size
        flipped_boxes = []
        for cls, x1, y1, x2, y2 in boxes:
            new_y1 = h - y2
            new_y2 = h - y1
            flipped_boxes.append((cls, x1, new_y1, x2, new_y2))
        return img, flipped_boxes

# -----------------------------------------
# Making sure scaling and pasting work well
# -----------------------------------------

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

# -------------------------------
# Transform 
# -------------------------------

# Light random augmentations
random_augmentations = [
    T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05),#(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    RandomGaussianBlur(),
    #RandomShadow(p=0.7),
    RandomBlackLines(p=0.5, max_lines=3, max_thickness=5)
]

# Randomly apply a subset of the defined augmentations
class RandomSubsetTransform:
    def __init__(self, transforms, num_choices=2):
        self.transforms = transforms
        self.num_choices = num_choices

    def __call__(self, img):
        chosen_transforms = random.sample(self.transforms, self.num_choices)
        for t in chosen_transforms:
            img = t(img)
        return img

    

def get_train_transform():

    return T.Compose([
    RandomSubsetTransform(random_augmentations, num_choices=2),
    T.ToTensor(),
    RandomGaussianNoise(0., 0.02, p=0.3),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# -------------------------------
#  CutPaste Chocolate Dataset
# -------------------------------

class ChocolateCutPasteDataset(Dataset):
    """
    A PyTorch dataset for chocolate images with YOLO-style annotations,
    supporting CutPaste augmentation and geometrical transforms (crop, flip),
    while keeping bounding boxes updated accordingly.
    """
    def __init__(self, image_dir, label_dir, transform=None, cutpaste_p=0.5,
                 crop_resize_with_boxes=None, flip_with_boxes=None, vflip_with_boxes=None,
                 min_size_ratio=0.05, max_iou=0.3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.transform = transform
        self.cutpaste_p = cutpaste_p
        self.crop_resize_with_boxes = crop_resize_with_boxes
        self.flip_with_boxes = flip_with_boxes
        self.vflip_with_boxes = vflip_with_boxes
        self.min_size_ratio = min_size_ratio  # filter out very small boxes
        self.max_iou = max_iou  # prevent donor crop from overlapping too much

    def __len__(self):
        return len(self.image_files)

    def load_labels(self, label_path, img_width, img_height):
        """
        Load bounding boxes from YOLO-format .txt file and convert to pixel x1y1x2y2.
        """
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
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        boxes = self.load_labels(label_path, width, height)  # → list of (cls, x1, y1, x2, y2)

        # --------- CutPaste augmentation ----------
        if random.random() < self.cutpaste_p and len(boxes) > 0:
            donor_idx = random.randint(0, len(self.image_files) - 1)
            donor_name = self.image_files[donor_idx]
            donor_img_path = os.path.join(self.image_dir, donor_name)
            donor_label_path = os.path.join(self.label_dir, donor_name.replace(".jpg", ".txt"))

            donor_img = Image.open(donor_img_path).convert("RGB")
            donor_width, donor_height = donor_img.size
            donor_boxes = self.load_labels(donor_label_path, donor_width, donor_height)

            if len(donor_boxes) > 0:
                cls, x1, y1, x2, y2 = random.choice(donor_boxes)
                donor_crop = donor_img.crop((x1, y1, x2, y2))

                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < width and crop_h < height:
                    paste_x = random.randint(0, width - crop_w)
                    paste_y = random.randint(0, height - crop_h)
                    donor_box_coords = (paste_x, paste_y, paste_x + crop_w, paste_y + crop_h)

                    if all(compute_iou((bx1, by1, bx2, by2), donor_box_coords) <= self.max_iou
                        for _, bx1, by1, bx2, by2 in boxes):
                        image.paste(donor_crop, (paste_x, paste_y))
                        boxes.append((cls, *donor_box_coords))  # ← just a 5-tuple

        # --------- Apply geometry-aware transforms ----------
        if self.crop_resize_with_boxes:
            image, boxes = self.crop_resize_with_boxes(image, boxes)
        if self.flip_with_boxes:
            image, boxes = self.flip_with_boxes(image, boxes)
        if self.vflip_with_boxes:
            image, boxes = self.vflip_with_boxes(image, boxes)

        # --------- Filter out tiny boxes ----------
        width, height = image.size
        boxes = filter_boxes_by_relative_size(boxes, width, height, self.min_size_ratio)

        # --------- Final additional transforms ----------
        if self.transform:
            image = self.transform(image)

        return image, boxes

    
    

    ##note:
    #to use tranformations that affect the boxes, you need to pass them as arguments to the dataset
    #ex:

    #cropper = RandomResizedCropWithBoxes(size=150)
    #flipper = RandomHorizontalFlipWithBoxes(p=0.5)
    #vflipper = RandomVerticalFlipWithBoxes(p=1.0) 

    #dataset = ChocolateCutPasteDataset(
    #    image_dir="...",
    #    label_dir="...",
    #    transform=train_transform_balanced,
    #    crop_resize_with_boxes=cropper,
    #    flip_with_boxes=flipper,
    #    vflip_with_boxes=vflipper,
    #    cutpaste_p=0.5
    #)