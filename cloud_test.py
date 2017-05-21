import sys
import numpy as np
import lmdb

from PIL import Image

sys.path.insert(0,'/mnt/eg5_data/caffe/python')
import caffe

caffe.set_mode_cpu()

train_path = "/home/eg4/toka/test1/train.txt"
test_path = "/home/eg4/toka/test1/test.txt"
image_path = "/mnt/eg4_data/Image_dataset/Output_dataset/test/"
path = "/home/eg4/toka"

instanceList = []
output_string = ""
correctLabels = 0
incorrectLabels = 0
positiveLabels = 0
negativeLabels = 0
positivePredictions = 0
negativePredictions = 0

#open train file
gt_file = open(test_path, "r")

# Store images in a list
while (True):
    line = gt_file.readline()
    # Check if we have reached the end
    if (len(line) == 0):
        break
    # Add the line to the list
    instanceList.append(line)

 # Load network
net = caffe.Classifier(path+'/bvlc_reference_caffenet/deploy.prototxt',
                       path+'/bvlc_reference_caffenet/mymodel_iter_3000.caffemodel',
                       mean=np.load('/mnt/eg4_data/Image_dataset/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       image_dims=(256, 256),
                       channel_swap=(2, 1, 0),
                       raw_scale=255)

# Loop through the ground truth file, predict each image's label and store the wrong ones
counter = 0
for instance in instanceList:
    values = instance.split()
    img_path = values[0]
    sentiment = int(values[1])

    # Load image
    im = caffe.io.load_image(image_path+img_path)

    # Make a forward pass and get the score
    prediction = net.predict([im])

    # Check if the prediction was correct or not
    if prediction[0].argmax() == sentiment:
        correctLabels += 1
    else:
        incorrectLabels += 1

    # Update label counter
    if sentiment == 0:
        negativeLabels += 1
    else:
        positiveLabels += 1

    # Update prediction counter (negative = 0, positive = 1)
    if prediction[0].argmax() == 0:
        negativePredictions += 1
    else:
        positivePredictions += 1

    counter += 1

gt_file.close()
accuracy = 100. * correctLabels / (correctLabels + incorrectLabels)

# Print accuracy results
print 'Accuracy = ', str(accuracy)
print '---------------------------------'

output_string += 'Positive images: {0}\n    Negative images: {1}\n    Positive predictions: {2}\n    Negative predictions: {3}\n'.format(
             str(positiveLabels), str(negativeLabels), str(positivePredictions), str(negativePredictions))

print output_string

