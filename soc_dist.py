import cv2
import torch
import torchvision
import torchvision.transforms as T 
import numpy as np


video_path = '/Users/varun/Documents/Deep_Learning/Learn_PYTORCH/ped_detect/video_org.mp4'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

(model.eval())

labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(image, threshold):

      transform = T.Compose([T.ToTensor()]) 
    
      img = transform(image) 
        
      pred = model([img])

      pred_class = [labels[i] for i in list(pred[0]['labels'].numpy())] 

      bb = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
     
      prob = list(pred[0]['scores'].detach().numpy())

      prob_t = [prob.index(x) for x in prob if x > threshold][-1]

      bb = bb[:prob_t+1]

      pred_class = pred_class[:prob_t+1]

      return bb, pred_class


def detect_person(image,threshold):
    
    boxes, pred_cls = get_prediction(image,0.5)
    # for i in range(len(boxes)):
    #     if(pred_cls[i]=='person'):
    #         cv2.rectangle(image, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=6) 
    
    return image


cam = cv2.VideoCapture(video_path)

while(True):
    ret,image = cam.read()
    imag = detect_person(image,0.5)
    cv2.imshow('img',imag)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()