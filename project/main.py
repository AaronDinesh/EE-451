import os
from utils import *



train_loader = yolo_dataloader("./data/train", image_size=(640, 640), collate_fn=collate_fn)

images, targets = next(iter(train_loader))
visualize_yolo_dataloader_batch(images, targets, class_names=['Amandina', 'Arabia', 'Comtesse', 'Creme_brulee', 'Jelly_Black', 'Jelly_Milk', 'Jelly_White', 'Noblesse', 'Noir_authentique', 'Passion_au_lait', 'Stracciatella', 'Tentation_noir', 'Triangolo'])