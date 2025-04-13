
import os
from ultralytics import YOLO

print(os.listdir("raw_page_images"))
dossiers=os.listdir("raw_page_images")
model=YOLO("yolov11n-face.pt")

nb_predictions=len(os.listdir("runs/detect"))+1


for dossier in dossiers:
    model(source="raw_page_images/"+dossier,save_txt=True)
    os.rename("runs/detect/predict"+str(nb_predictions)+"/labels","runs/detect/predict"+str(nb_predictions)+"/"+dossier)

model=YOLO("yolo11n.pt")
nb_predictions+=1

for dossier in dossiers:
    model(source="raw_page_images/"+dossier,save_txt=True)
    os.rename("runs/detect/predict"+str(nb_predictions)+"/labels","runs/detect/predict"+str(nb_predictions)+"/"+dossier)