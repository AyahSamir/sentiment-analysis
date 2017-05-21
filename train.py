import sys
import numpy as np
import lmdb
import cv2
from PIL import Image

#sys.path.insert(0,'/home/ayah/caffe/python')
import caffe

input_path= "/Agg_AMT_Candidates/"
output_path= "/train_resized/"
instanceList = []
gt_file = open('/train.txt', "r")

while (True):
    line = gt_file.readline()
    # Check if we have reached the end
    if (len(line) == 0):
        break
    # Add the line to the list
    instanceList.append(line)


for instance in instanceList:
    values = instance.split()
    image_path = values[0]
    input_image = cv2.imread(input_path+image_path)
    resized_image = cv2.resize(input_image, (256, 256))
    cv2.imwrite(output_path+image_path,resized_image)
