#If you use YOLO11++ model from this repository in your work, please cite it using the following format:

#BibTeX
#@article{hel25YOLO11++UAES,
#  title={Universal Autonomous Driving System: Innovation with YOLO11++-based Universal Autonomous Expert System},
#  author={Maher, Helaoui and Sahbi, Bahroun and Ezzeddine, Zagrouba},
#  journal={The Visual Computer (Submited)},
#  url = {[https://github.com/ultralytics/ultralytics](https://github.com/maherhelaoui/YOLO11pp-based-UAES/)},
#  year={2025},
#  publisher={Springer}
#}
#Please note that the DOI will be added to the citation once it is available. 



import os
import shutil
from pathlib import Path
from ultralytics.utils.benchmarks import RF100Benchmark
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque


# YOLO11 is trained on coco dataset.
#Train YOLO11++ on different Datasets 

model = YOLO("yolo11n.yaml")
# Charger les modèles YOLO
#model3 = YOLO("yolo11++MP.pt")  # if model.train(data="medical-pills.yaml", epochs=100)
#model3 = YOLO("yolo11++GW.pt")  # if model.train(data="GlobalWheat2020.yaml", epochs=100)
#model3 = YOLO("yolo11++S.pt")  # if model.train(data="signature.yaml", epochs=100)
#model3 = YOLO("yolo11++AW.pt")  # if model.train(data="african-wildlife.yaml", epochs=100)
#model3 = YOLO("yolo11++BT.pt")  # if model.train(data="brain-tumor.yaml", epochs=100)
model3 = YOLO("yolo11++CC128.pt")  # if model.train(data="coco128.yaml", epochs=100)
model = model3



#results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640) # Train MP Dataset use model yolo11++MP.pt
#results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640) # Train GW Dataset use model yolo11++GW.pt
#results = model.train(data="signature.yaml", epochs=100, imgsz=640) # Train SD Dataset use model yolo11++S.pt
#results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640) # Train AW Dataset use model yolo11++AW.pt
#results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640) # Train BT Dataset use model yolo11++BT.pt
results = model.train(data="coco128.yaml", epochs=100) # Train CC128 Dataset use model yolo11++CC128.pt 






