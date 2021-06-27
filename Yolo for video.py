# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:21:46 2021

@author: nehal
"""

import cv2
import numpy as np
import os


# Initializing Yolo Frame Work
net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

#Storing class names   
classes=[]
with open('coco.names','r') as f:
    classes=f.read().splitlines()

input_number=3
cap=cv2.VideoCapture('Test'+str(input_number)+'.mp4')


while True:

    _,img=cap.read()
    
    if img is None:
        break
    else:
        height,width,_=img.shape
  
    # 1/255 -Scaling
    # (448,448) - Size of image
    # (0,0,0) - No Mean Subtraction
    # swapRB=True  -To convert BGR to RGB
    # crop = False - To avoid cropping
    
    # Blob is done because the model accepts a 4D input
    blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    
            
    # Giving blob as input to network    
    net.setInput(blob)
    # Getting the name of the output layers
    #  To pass in the next function to get the output of these layers
    output_layers_names=net.getUnconnectedOutLayersNames()    
    # Getting the output of the yolo network
    layerOutputs=net.forward(output_layers_names)
    

    
    # To store bounding boxes,confidences,class_ids
    boxes=[]
    confidences=[]
    class_ids=[]
    
    # To extract and store above mentioned information from the Output layers of the yolo network (layerOutputs)
    for output in layerOutputs:
        for detection in output:
            # Storing class probabilities
            scores=detection[5:]           
            # Storing Highest score location
            class_id=np.argmax(scores)
            # Storing confidence/probabilty of detected object
            confidence=scores[class_id]
            
            # Storing bounding box cordinares of objects with probabilty match of 75% and above
            if ((confidence > 0.75) & (classes[class_id]=='person' or classes[class_id]=='motorbike')):
                
                # Multiplying with width to rescale to size of original image
               
                center_x=int(detection[0]*width)              
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                
                # Getting Position of upper left corner
                x=int(center_x-w/2)
                y=int(center_y-h/2)
                
                # Storing bounding box coordinates, confidences and class_id
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
                
                
    
    # Checking how many boxes where detected
    #print(len(boxes))
    
    # Surpressing Common detected boxes
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.75,0.65)
    
    # If images are detected in frame
    if (len(indexes)!=0):
        detected_obj=indexes.flatten()
        
    
    
        font=cv2.FONT_HERSHEY_PLAIN
        
        # Assigning color for each detected object
        # 3 is the number of channels
        colors=np.random.uniform(0,255,size=(len(boxes),3))
        
        count_p=0
        count_b=0
        for i in detected_obj:

            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            confidence=str(round(confidences[i],2))
            color=colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,label+" "+confidence, (x,y+20),font,2,(255,255,255),2)
            print(label,confidence)
            
            if label in 'person':
                count_p=count_p+1
            elif label in 'motorbike':
                count_b=count_b+1
        
        #print(count_p,count_b)
        
        if ((count_p>=3) & (count_b>=1)):
            os.chdir('C:\\Users\\nehal\\Music\\11.Deep Learning\\Project\\Final Project\\Violaters')
            cv2.imwrite('Violater'+str(input_number)+'.jpg', img)
            print('Violation detected !!  Save this image .')
        else:
            print('No violation detected.')
            
        
        cv2.imshow('Image',img)
        key=cv2.waitKey(1)
    
    # If no images detected in frame
    else:
        cv2.imshow('Image',img)
        key=cv2.waitKey(1)
    
    
    
    # To Escape from video on click ESC
    # 27 refers to escape key
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()

