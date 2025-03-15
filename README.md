# Universal Autonomous Turing Machine Based yolo11++ detector and UAES For Driving System 

____________________________________________________________________________________________________________________

                                 
                                    A. yolo11++ proposed detector                               


____________________________________________________________________________________________________________________

To train and test the proposed detector yolo11++ with the different studied datasets you can use TrainYolo11++.ipynb 
available in this repository. It is tested on Colab.
Or you can use TrainYOLO11++.py file  
YOLO11++ needs installing

pip install ultralytics  # For YOLO11++ like YOLO11 (https://docs.ultralytics.com/quickstart/#install-ultralytics)
_____________________________________________________________________________________________________________________
                                        
                                        Download trained models

_____________________________________________________________________________________________________________________

We prepared for each Dataset a trained model of yolo11++ as proposed in the Visual computer paper. 
We present a best trained model of yolo11++ for each studied Dataset.

Trained Model 1. yolo11++MP.pt : Trained model of yolo11++ for Medical-Pills MP Dataset
Trained Model 2. yolo11++GW.pt : Trained model of yolo11++ for GlobalWheat2020 GW Dataset
Trained Model 3. yolo11++S.pt : Trained model of yolo11++ for Signature S Dataset
Trained Model 4. yolo11++AW.pt : Trained model of yolo11++ for African-Wildlife AW Dataset
Trained Model 5. yolo11++BT.pt : Trained model of yolo11++ for Brain-Tumor BT Dataset

______________________________________________________________________________________________________________________
 
                                 Train YOLO11++ on different Datasets i
______________________________________________________________________________________________________________________
 

# For each Dataseti we charge the trained YOLO++ Models
#model3 = YOLO("yolo11++MP.pt")  # if model.train(data="medical-pills.yaml", epochs=100)
#model3 = YOLO("yolo11++GW.pt")  # if model.train(data="GlobalWheat2020.yaml", epochs=100)
#model3 = YOLO("yolo11++S.pt")  # if model.train(data="signature.yaml", epochs=100)
#model3 = YOLO("yolo11++AW.pt")  # if model.train(data="african-wildlife.yaml", epochs=100)
#model3 = YOLO("yolo11++BT.pt")  # if model.train(data="brain-tumor.yaml", epochs=100)
model3 = YOLO("yolo11++CC128.pt")  # if model.train(data="coco128.yaml", epochs=100)
model = model3


# Train chargedYOLO Model on each Dataseti
#results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640) # Train MP Dataset use model yolo11++MP.pt
#results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640) # Train GW Dataset use model yolo11++GW.pt
#results = model.train(data="signature.yaml", epochs=100, imgsz=640) # Train SD Dataset use model yolo11++S.pt
#results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640) # Train AW Dataset use model yolo11++AW.pt
#results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640) # Train BT Dataset use model yolo11++BT.pt
results = model.train(data="coco128.yaml", epochs=100) # Train CC128 Dataset use model yolo11++CC128.pt 



To Train YOLO11++ on different Datasets you can use 

$ python TrainYOLO11++.py
or
$ python3 TrainYOLO11++.py

You can charge the trained YOLO++ Models by selecting one from this list in TrainYOLO11++.py
#model3 = YOLO("yolo11++MP.pt")  # if model.train(data="medical-pills.yaml", epochs=100)
#model3 = YOLO("yolo11++GW.pt")  # if model.train(data="GlobalWheat2020.yaml", epochs=100)
#model3 = YOLO("yolo11++S.pt")  # if model.train(data="signature.yaml", epochs=100)
#model3 = YOLO("yolo11++AW.pt")  # if model.train(data="african-wildlife.yaml", epochs=100)
#model3 = YOLO("yolo11++BT.pt")  # if model.train(data="brain-tumor.yaml", epochs=100)
model3 = YOLO("yolo11++CC128.pt")  # if model.train(data="coco128.yaml", epochs=100)
model = model3


You can change the used dataset by selecting one from this list in TrainYOLO11++.py

#results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640) # Train MP Dataset use model yolo11++MP.pt
#results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640) # Train GW Dataset use model yolo11++GW.pt
#results = model.train(data="signature.yaml", epochs=100, imgsz=640) # Train SD Dataset use model yolo11++S.pt
#results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640) # Train AW Dataset use model yolo11++AW.pt
#results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640) # Train BT Dataset use model yolo11++BT.pt
results = model.train(data="coco128.yaml", epochs=100) # Train CC128 Dataset use model yolo11++CC128.pt 

Each yolo11++ model is generated as decribed in Machine Vision and Applications journal paper.


______________________________________________________________________________________________________________________

                             B Real Word Simplified Demo of Universal Autonomous Driving System 
______________________________________________________________________________________________________________________
#Tor run this demonstartion of UADS you can use main.ipynb on Colab. Or Demo.py on your machine
# You can download VIDS18.mp4 from Real word proposed dataset available in https://drive.google.com/file/d/1nb-DZDMq62G_w_xrNAZzeS-oNx4GxOAC/view?usp=drive_link
# Please add VIDS18.mp4 to your work space on Colab or your machine

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
#make a trained model on different datasets is it importants ?
#How improve yolo 11 ?
# YOLO11 is trained on coco dataset.
# what is the result if we concatinate a multi trained models ?
#Create new model by generate boxes, scores, classes as concatenation of 3 boxes, scores, classes
#Create a concatenation of 3 different models results

model = YOLO("yolo11n.yaml")
model1 = YOLO("yolov8sign.pt")  # detect all sign. You can use any sign detector Model. The model chosed in this demo is mensioned in the Visual computer journal paper. It must be added to the work space. 
model = model1
model2 = YOLO("yolov8nbt.pt")  # detect traffic Light red green yellow. You can use any traffic Light detector Model. The model chosed in this demo is mensioned in the Visual computer journal paper. It must be added to the work space. 
model = model2
model3 = YOLO("yolo11++CC128.pt")  # This traned model is the most adequate to detect cars bus trained on Coco128 Dataset 
model = model3



x1=1000
# Open video
video_path = "VIDS18.mp4"
cap = cv2.VideoCapture(video_path)

# Detect speed demo
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('VIDS/VIDSVDMD13.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# traked objects
tracked_objects = {}

# Compute speed demo
def calculate_speed(prev_position, current_position, fps):
    distance = np.linalg.norm(np.array(current_position) - np.array(prev_position))
    speed = distance * fps  # en pixels par seconde
    return speed

# Read traffic Signs (simplified)
def read_traffic_signs(frame):
    # Ici, vous pouvez ajouter un modèle de détection de panneaux de signalisation
    # Pour cet exemple, nous allons simplement retourner un panneau fictif
    for r in results3:
       for box in r.boxes:
          if x1<5 :
              cv2.putText(frame, "Recommend: Stop object ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          else :
               if x1<20 :
                     cv2.putText(frame, "Recommend: Worning object ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
               else :
                cv2.putText(frame, "Recommend: Keep Line", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
         
    for r in results2:
       for box in r.boxes:
          print(int(box.cls))
          if(int(box.cls)==0):
                 cv2.putText(frame, "Recommend:           Stop red", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          if(int(box.cls)==1):
                 cv2.putText(frame, "Recommend:            Green Go", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for r in results1:
       #print(model1.names)
       for box in r.boxes:
          #print(box.cls)) 
          if(int(box.cls)):
             print(int(box.cls))
             #cv2.putText(frame, f"                               id :{int(box.cls)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    

# Demo display recommendations (simplified)
def display_recommendations(frame, sign):
    if sign == "Go":
        cv2.putText(frame, "Recommendation: Green Go", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if sign == "Stop":
        cv2.putText(frame, "Recommendation: Stoooooop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #else :
        #cv2.putText(frame, "Recommendation: Keep Line", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Demo video treatement
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des objets avec YOLO
    results1 = model1(frame)
    for r in results1:
        frame = r.plot()
    results2 = model2(frame)
    for r in results2:
        frame = r.plot()

    results3 = model3(frame)
    # Demo resultats treatement
    for result in results3:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == 2:  # Class ID 2 correspond aux voitures dans YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_id = f"{x1}_{y1}_{x2}_{y2}"

                # Demo compute speed (simplified)
                if object_id in tracked_objects:
                    prev_position = tracked_objects[object_id]
                    current_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                    speed = calculate_speed(prev_position, current_position, fps)
                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                tracked_objects[object_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Dessiner la boîte autour de la voiture
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Read_traffic_signs
    sign = read_traffic_signs(frame)

    # display_recommendations
    display_recommendations(frame, sign)

    # Output Video
    output_video.write(frame)

    # Display frame (optionnel)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free ressources
cap.release()
output_video.release()
cv2.destroyAllWindows()





If you use this demo or Universal Autonomous Driving System software from this repository in your work, please cite it using the following format:
____________________________________________________________________________________________________________________
                                       
                                              BibTeX
                                              
____________________________________________________________________________________________________________________

@article{hel25YOLO11++UAES, 
title={Universal Autonomous Turing Machine Based YOLO11++ detector and Universal Autonomous Expert System for Driving System}, author={Maher, Helaoui and Sahbi, Bahroun and Ezzeddine, Zagrouba}, journal={Machine Vision and Applications (Submitted)}, url = {https://github.com/maherhelaoui/YOLO11pp-based-UAES/}, year={2025}, publisher={Springer} }



