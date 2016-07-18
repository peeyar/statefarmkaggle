# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 23:43:02 2016

@author: anandpreshob
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 22:03:30 2016

@author: anandpreshob
"""
#%% import libraries
import cv2, os, time, glob
import tensorflow as tf
from sklearn import svm
from sklearn import datasets
import numpy as np
#%%
import tensorflow.contrib.learn as skflow
#%% function for loading image
img_size = 50
sz = (img_size, img_size)

def process_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    return img
#%%
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')    
#%%
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
#    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

#%% Load images
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
x=X_train
y=Y_train
#%% initialise a training model
# Store layers weight & bias
keep_prob=0.75
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
#    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
#    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
#%%
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#%%
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"









#clf = svm.SVC()
#dataset_size = len(X_train)
#x3d = X_train.reshape(dataset_size,-1)
#iris = datasets.load_iris()
#X, y = iris.data, iris.target
#clf.fit(X_train, Y_train)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
#%% convert the loaded image list into numpy array
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
 