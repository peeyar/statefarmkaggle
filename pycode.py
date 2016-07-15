# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 22:03:30 2016

@author: anandpreshob
"""
#%%
import cv2, os, time, glob, tensorflow
from sklearn import svm
from sklearn import datasets
import numpy as np
#%%
import tensorflow.contrib.learn as skflow
#%%
img_size = 50
sz = (img_size, img_size)

def process_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    return img
#%%
start = time.time()

X_train = []
Y_train = []
for j in range(10):
    print('Load folder c{}'.format(j))
    path = os.path.join('/home/anandpreshob/statefarm/imgs', 'train', 'c' + str(j), '*.jpg')
    files = glob.glob(path)
    print(files)
    for fl in files:
        flbase = os.path.basename(fl)
        img = process_image(fl)
        X_train.append(img)
        Y_train.append(j)
end = time.time() - start
print("Time: %.2f seconds" % end)
#%%
clf = svm.SVC()
dataset_size = len(X_train)
x3d = X_train.reshape(dataset_size,-1)
#iris = datasets.load_iris()
#X, y = iris.data, iris.target
#clf.fit(X_train, Y_train)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
#%%
x=np.asarray(X_train)
y=np.asarray(Y_train)
#%%
classifier = skflow.TensorFlowDNNClassifier(
...     hidden_units=[10, 20, 10], 
...     n_classes=2, 
...     batch_size=128, 
...     steps=500, 
...     learning_rate=0.05)
classifier.fit(x,y)
#%%
xtest=process_image('/home/anandpreshob/statefarm/imgs/test/img_24.jpg')
print(classifier.predict(xtest))







#%%
# path = os.path.join('/media/anandpreshob/New Volume/Anand/MLCV/Kaggle/Statefarm', 'train')
 pfiles = glob.glob('/media/anandpreshob/New Volume/Anand/MLCV/Kaggle/Statefarm/imgs/train/c0/*.jpg')
 