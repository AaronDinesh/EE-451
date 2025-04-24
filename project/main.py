import pandas as pd
from check import IDS, COLS, check_df  # Import everything needed from check.py
from ultralytics import YOLO
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def TRAIN(model_config: str, params: dict):
    DATA_YAML_PATH = "data.yaml"
    model = YOLO(model_config) 
    model.train(
        data=DATA_YAML_PATH,
        epochs=params["epochs"],          
        imgsz=params["imgsize"],           
        batch=params["batch"],             
        patience=params["patience"],         
        workers=params["workers"],           
        name=params["name"],
        pretrained=False
    )

    return model

def USE_MODEL_ON_IMAGES(number_of_images_selected: int, model):
    test_dir = "test"
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    sampled_paths = random.sample(image_paths, number_of_images_selected)

    # Inference and display
    for img_path in sampled_paths:
        results = model(img_path,verbose=False)
        res_plotted = results[0].plot()  # Visualized prediction
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

def CREATE_TABLE(model):
    YOLO_NAMES = [
        "Amandina", "Arabia", "Comtesse", "Creme_brulee", "Jelly_Black",   
        "Jelly_Milk",
        "Jelly_White", "Noblesse", "Noir_authentique", "Passion_au_lait",
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

    # Get test image paths
    test_dir = "test"
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Map image filename to numeric ID
    image_name_to_id = {
        os.path.splitext(f)[0].lstrip("L"): int(os.path.splitext(f)[0].lstrip("L"))
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    # Inference results list
    all_rows = []
    total = len(image_paths)

    for index, img_path in enumerate(image_paths):
        results = model(img_path, verbose=False)[0]
        counts = {col: 0 for col in COLS}

        for cls_id in results.boxes.cls.tolist():
            yolo_class_name = YOLO_NAMES[int(cls_id)]
            target_col = YOLO_TO_COL[yolo_class_name]
            counts[target_col] += 1

        file_id = int(os.path.splitext(os.path.basename(img_path))[0].lstrip("L"))

        row = {"id": file_id}
        row.update(counts)
        all_rows.append(row)

        if((index + 1)%10==0):
            print(f"{(index + 1) / total * 100:.2f}% is finished")

    # Create DataFrame, fill missing IDs with 0s
    df = pd.DataFrame(all_rows)
    df = df.set_index("id")
    df = df.reindex(IDS).fillna(0).astype(int).reset_index()

    # Save CSV
    output_path = "submission.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved formatted CSV to {output_path}")

    # Optional: Validate
    check_df(df, df_name="Generated Submission")
    return df

