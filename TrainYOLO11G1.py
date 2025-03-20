#If you use YOLO11Gi model from this repository in your work, please cite it using the following format:

#BibTeX
#@article{hel25YOLO11G1ANES,
#  title={Universal Autonomous Driving System: Innovation with YOLO11++-based Universal Autonomous Expert System},
#  author={Maher, Helaoui and Sahbi, Bahroun and Ezzeddine, Zagrouba},
#  journal={Neural Computing and Applications (Submited)},
#  url = {https://github.com/maherhelaoui/UATM-Based-yolo11--UAES/},
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
#Train YOLO11G1 on different Datasets => Obtain YOLO11G2 

model = YOLO("yolo11n.yaml")
# Charger les modèles YOLO
#model3 = YOLO("yolo11G1MP.pt")  # if model.train(data="medical-pills.yaml", epochs=100)
#model3 = YOLO("yolo11G1GW.pt")  # if model.train(data="GlobalWheat2020.yaml", epochs=100)
#model3 = YOLO("yolo11G1S.pt")  # if model.train(data="signature.yaml", epochs=100)
#model3 = YOLO("yolo11G1AW.pt")  # if model.train(data="african-wildlife.yaml", epochs=100)
#model3 = YOLO("yolo11G1BT.pt")  # if model.train(data="brain-tumor.yaml", epochs=100)
model3 = YOLO("yolo11G1CC128.pt")  # if model.train(data="coco128.yaml", epochs=100)
model = model3



#results = model.train(data="medical-pills.yaml", epochs=30, imgsz=640) # Train MP Dataset use model yolo11G1MP.pt => obtain yolo11G2MP.pt
#results = model.train(data="GlobalWheat2020.yaml", epochs=30, imgsz=640) # Train GW Dataset use model yolo11G1GW.pt => obtain yolo11G2GW.pt
#results = model.train(data="signature.yaml", epochs=30, imgsz=640) # Train SD Dataset use model yolo11G1S.pt => obtain yolo11G2S.pt
#results = model.train(data="african-wildlife.yaml", epochs=30, imgsz=640) # Train AW Dataset use model yolo11G1AW.pt => obtain yolo11G2AW.pt 
#results = model.train(data="brain-tumor.yaml", epochs=30, imgsz=640) # Train BT Dataset use model yolo11G1BT.pt => obtain yolo11G2BT.pt 
results = model.train(data="coco128.yaml", epochs=30) # Train CC128 Dataset use model yolo11G1CC128.pt => obtain yolo11G2CC128.pt 






