
def get_images(img_file, number):
    f = open(img_file, "rb") # Open file in binary mode
    f.read(16) # Skip 16 bytes header
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    return images

def get_labels(label_file, number):
    l = open(label_file, "rb") # Open file in binary mode
    l.read(8) # Skip 8 bytes header
    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
    return labels

import os
import numpy as np
from numpy import uint8
from skimage import io

def convert_png(images, labels, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for i in range(len(images)):
        out = os.path.join(directory, "%06d-num%d.png"%(i,labels[i]))
        io.imsave(out, np.array(images[i],dtype=np.uint8).reshape(28,28))

def output_csv(images, labels, out_file):
    o = open(out_file, "w")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()
number = 100
train_images = get_images("D:/hocmay/mnist/train-images-idx3-ubyte", number)
train_labels = get_labels("D:/hocmay/mnist/train-labels-idx1-ubyte", number)
convert_png(train_images, train_labels, "preview")

def output_csv(images, labels, out_file):
    o = open(out_file, "w")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()

from mnist import MNIST
mndata = MNIST('./mnist')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
print(train_labels)

import numpy
train_images = numpy.array(train_images)/255
import numpy as np
from sklearn import svm, metrics

print("TRAIN")
TRAINING_SIZE = 10000
train_images = get_images("mnist/train-images-idx3-ubyte", TRAINING_SIZE)
train_images = np.array(train_images)/255
train_labels = get_labels("mnist/train-labels-idx1-ubyte", TRAINING_SIZE)

clf = svm.SVC(C=100)
clf.fit(train_images, train_labels)
TEST_SIZE = 500
test_images = get_images("mnist/t10k-images-idx3-ubyte", TEST_SIZE)
test_images = np.array(test_images)/255
test_labels = get_labels("mnist/t10k-labels-idx1-ubyte", TEST_SIZE)

print("PREDICT")
predict = clf.predict(test_images)

print("RESULT")
ac_score = metrics.accuracy_score(test_labels, predict)
cl_report = metrics.classification_report(test_labels, predict)
print("Score = ", ac_score)
print(cl_report)
from joblib import dump, load
dump(clf, 'mnist-svm.joblib')



