from src.utils import *
import pandas as pd
from src.models import YOLOv8Lite, TinyYOLO, CompactYOLOv2
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


IMG_PARAM = {}
IMG_PARAM["IMG_SIZE"] = 640
IMG_PARAM["ANCHOR"] = 3
IMG_PARAM["NUM_CLASSES"] = 13

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv8Lite().to(device)

if(isinstance(model,TinyYOLO) or isinstance(model,CompactYOLOv2)):
    IMG_PARAM["GRID_SIZE"] = 20
elif(isinstance(model,YOLOv8Lite)):
    IMG_PARAM["GRID_SIZE"] = 40


current_dir = os.getcwd()
train_root_dir = Path(f'{current_dir}/data/train')

# Use in your dataset
train_dataset = YoloGridDataset(
    root_dir=train_root_dir,
    GRID_SIZE=IMG_PARAM["GRID_SIZE"],
    image_size=(IMG_PARAM["IMG_SIZE"], IMG_PARAM["IMG_SIZE"])
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 1
name_of_saved_pt = "YoloV8Main2"


test_root_dir = Path(f'{current_dir}/data/test')

test_dataset = YoloGridDataset(
    root_dir=test_root_dir,
    GRID_SIZE=IMG_PARAM["GRID_SIZE"],
    image_size=(IMG_PARAM["IMG_SIZE"], IMG_PARAM["IMG_SIZE"])
)

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)


print_total_paramters(model)


model_save_path = f"{current_dir}/model_weights"
model ,metrics, best_f1_model = train_model(model=model, total_epochs=EPOCHS, optimizer=optimizer, device=device, 
            per_epoch_save=40, train_loader=train_loader, test_loader=test_loader, plotting_callback=None, 
            name_of_saved_pt=name_of_saved_pt, pt_save_path=model_save_path, IMG_PARAM = IMG_PARAM)

print("-----------------------------------------------------------------------------------------------------")
print("---------------------------------------- END OF THE TRAINING ----------------------------------------")
print("-----------------------------------------------------------------------------------------------------")

list_of_threshold = np.array(range(1,26))*0.01
list_of_model_names = [f"{name_of_saved_pt}_best_f1_from_training.pt"]
test_root_dir = Path(f'{current_dir}/data')
#epoch 200 conf 13 dist 40 -> 85

current_best_f1 = -np.inf

for model_name in list_of_model_names:
    for threshold in list_of_threshold:

        df = create_table_from_pt_nms_for_training_F1(
            model = best_f1_model,
            IMG_PARAM = IMG_PARAM,
            conf_threshold = threshold,
            min_dist = 40
        )

        current_f1 = evaluate_f1_score_test(df)

        if(current_f1 > current_best_f1):
            current_best_f1 = current_f1
            output_path = f"submission_BEST.csv"
            df.to_csv(output_path, index=False)

print("-----------------------------------------------------------------------------------------------------")
print("------------------------------------ END OF THE CONFIDENCE TUNING -----------------------------------")
print("-----------------------------------------------------------------------------------------------------")


print("F1 score obtained:")
print(current_best_f1)



# Ensure the folder for saving graphs exists
os.makedirs("graphs", exist_ok=True)

name = "YoloV8Main"

# Unpacking metrics
epochs = metrics["epoch"]
train_loss = metrics["train_loss"]
test_loss = metrics["test_loss"]
mAP50_training = metrics["mAP50_training"]
mAP50_testing = metrics["mAP50_testing"]
testing_F1 = metrics["testing-F1"]

# Plot 1: Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"graphs/{name}_loss_graph.png")
plt.close()

# Plot 2: mAP50 Graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, mAP50_training, label='mAP50 Training')
plt.plot(epochs, mAP50_testing, label='mAP50 Testing')
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.title('mAP50 Training vs Testing')
plt.legend()
plt.grid(True)
plt.savefig(f"graphs/{name}_mAP50_graph.png")
plt.close()

# Plot 3: Testing F1 Score
plt.figure(figsize=(8, 5))
plt.plot(epochs, testing_F1, label='Testing F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Testing F1 Score over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f"graphs/{name}_testing_f1_graph.png")
plt.close()

print(f"Graphs have been saved in the '{current_dir}/graphs' directory.")
