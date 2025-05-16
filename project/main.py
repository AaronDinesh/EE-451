from src.utils import *
import numpy as np
from pathlib import Path
from src.models import YOLOv8Lite

current_dir = os.getcwd()
threshold = 0.11
model_name = "CURRENT_BEST.pt"
test_root_dir = Path(f'{current_dir}/data')

IMG_PARAM = {}
IMG_PARAM["IMG_SIZE"] = 640
IMG_PARAM["ANCHOR"] = 3
IMG_PARAM["NUM_CLASSES"] = 13
IMG_PARAM["GRID_SIZE"] = 40


df = create_table_from_pt_nms(data_root_dir=test_root_dir, 
                                      model_path=f"{current_dir}/model_weights/{model_name}",
                                      model_class=YOLOv8Lite,
                                      IMG_PARAM=IMG_PARAM,
                                      conf_threshold=threshold,
                                      min_dist=40)
