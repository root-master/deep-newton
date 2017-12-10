import tensorflow as tf

tf.reset_default_graph()

import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)

import numpy as np
import sys
import collections

import pickle
import pandas as pd

from sklearn.cluster import KMeans
from numpy import linalg as LA

import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
        '--storage', '-m', default=20,
        help='The Memory Storage')
parser.add_argument(
        '--mini_batch', '-batch', default=1024,
        help='minibatch size')
parser.add_argument(
        '--whole_gradient', action='store_true',default=False,
        help='Compute the gradient using all data')
args = parser.parse_args()

print('----------------------------------------------')
print('L-BFGS')
print('----------------------------------------------')

print('----------------------------------------------')
print('architecture: LeNet-5 --- Data Set: MNIST')
print('----------------------------------------------')

# input and output shape
n_input   = data.train.images.shape[1]  # here MNIST data input (28,28)
n_classes = data.train.labels.shape[1]  # here MNIST (0-9 digits)

###############################################################################
########################## HYPER PARAMETER FOR LBFGS ##########################
###############################################################################
# memory limit
m = int(args.storage)
# minibatch size
minibatch = int(args.mini_batch)
# use entire data to compute gradient
use_whole_data_for_gradient = args.whole_gradient

# number of weights and bias in each layer
n_W = {}
dim_W = {}

# network architecture hyper parameters
input_shape = [-1,28,28,1]
W0 = 28
H0 = 28

# Layer 1 -- conv
D1 = 1
F1 = 5
K1 = 20
S1 = 1
W1 = (W0 - F1) // S1 + 1
H1 = (H0 - F1) // S1 + 1
conv1_dim = [F1, F1, D1, K1]
conv1_strides = [1,S1,S1,1] 
n_W['w_conv1'] = F1 * F1 * D1 * K1
n_W['b_conv1'] = K1 
dim_W['w_conv1'] = [F1, F1, D1, K1]
dim_W['b_conv1'] = [K1]
# Layer 2 -- max pool
D2 = K1
F2 = 2
K2 = D2
S2 = 2
W2 = (W1 - F2) // S2 + 1
H2 = (H1 - F2) // S2 + 1
layer2_ksize = [1,F2,F2,1]
layer2_strides = [1,S2,S2,1]

# Layer 3 -- conv
D3 = K2
F3 = 5
K3 = 50
S3 = 1
W3 = (W2 - F3) // S3 + 1
H3 = (H2 - F3) // S3 + 1
conv2_dim = [F3, F3, D3, K3]
conv2_strides = [1,S3,S3,1] 
n_W['w_conv2'] = F3 * F3 * D3 * K3
n_W['b_conv2'] = K3 
dim_W['w_conv2'] = [F3, F3, D3, K3]
dim_W['b_conv2'] = [K3]

# Layer 4 -- max pool
D4 = K3
F4 = 2
K4 = D4
S4 = 2
W4 = (W3 - F4) // S4 + 1
H4 = (H3 - F4) // S4 + 1
layer4_ksize = [1,F4,F4,1]
layer4_strides = [1,S4,S4,1]


# Layer 5 -- fully connected
n_in_fc = W4 * H4 * D4
n_hidden = 500
fc_dim = [n_in_fc,n_hidden]
n_W['w_fc'] = n_in_fc * n_hidden
n_W['b_fc'] = n_hidden
dim_W['w_fc'] = [n_in_fc,n_hidden]
dim_W['b_fc'] = [n_hidden]
# Layer 6 -- output
n_in_out = n_hidden
n_W['w_out'] = n_hidden * n_classes
n_W['b_out'] = n_classes
dim_W['w_out'] = [n_hidden,n_classes]
dim_W['b_out'] = [n_classes]


for key, value in n_W.items():
	n_W[key] = int(value)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    # 5 x 5 convolution, 1 input image, 20 outputs
    'w_conv1': tf.get_variable('w_conv1', shape=[F1, F1, D1, K1],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			initializer=tf.contrib.layers.xavier_initializer()),
           			#initializer=tf.zeros_initializer()),
    # 'conv1': tf.Variable(tf.random_normal([F1, F1, D1, K1])),
    # 5x5 conv, 20 inputs, 50 outputs 
    #'conv2': tf.Variable(tf.random_normal([F3, F3, D3, K3])),
    'w_conv2': tf.get_variable('w_conv2', shape=[F3, F3, D3, K3],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			initializer=tf.contrib.layers.xavier_initializer()),
           			#initializer=tf.zeros_initializer()),
    # fully connected, 800 inputs, 500 outputs
    #'fc': tf.Variable(tf.random_normal([n_in_fc, n_hidden])),
    'w_fc': tf.get_variable('w_fc', shape=[n_in_fc, n_hidden],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			initializer=tf.contrib.layers.xavier_initializer()),
           			#initializer=tf.zeros_initializer()),
    # 500 inputs, 10 outputs (class prediction)
    #'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    'w_out': tf.get_variable('w_out', shape=[n_hidden, n_classes],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			initializer=tf.contrib.layers.xavier_initializer()),
           			#initializer=tf.zeros_initializer()),
    'b_conv1': tf.get_variable('b_conv1', shape=[K1],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			 #initializer=tf.zeros_initializer()),
           			initializer=tf.contrib.layers.xavier_initializer()),
    'b_conv2': tf.get_variable('b_conv2', shape=[K3],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.zeros_initializer()),
           			initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc': tf.get_variable('b_fc', shape=[n_hidden],
           			#initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.zeros_initializer()),
           			initializer=tf.contrib.layers.xavier_initializer()),
    'b_out': tf.get_variable('b_out', shape=[n_classes],
           			#initializer=tf.random_normal_initializer(stddev = 0.01))
           			#initializer=tf.zeros_initializer()) 
           			initializer=tf.contrib.layers.xavier_initializer())
}

def model(x,_W):
	# Reshape input to a 4D tensor 
    x = tf.reshape(x, shape = input_shape)
    # LAYER 1 -- Convolution Layer
    conv1 = tf.nn.relu(tf.nn.conv2d(input = x, 
    								filter =_W['w_conv1'],
    								strides = [1,S1,S1,1],
    								padding = 'VALID') + _W['b_conv1'])
    # Layer 2 -- max pool
    conv1 = tf.nn.max_pool(	value = conv1, 
    						ksize = [1, F2, F2, 1], 
    						strides = [1, S2, S2, 1], 
    						padding = 'VALID')

    # LAYER 3 -- Convolution Layer
    conv2 = tf.nn.relu(tf.nn.conv2d(input = conv1, 
    								filter =_W['w_conv2'],
    								strides = [1,S3,S3,1],
    								padding = 'VALID') + _W['b_conv2'])
    # Layer 4 -- max pool
    conv2 = tf.nn.max_pool(	value = conv2 , 
    						ksize = [1, F4, F4, 1], 
    						strides = [1, S4, S4, 1], 
    						padding = 'VALID')
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.contrib.layers.flatten(conv2)
    fc = tf.nn.relu(tf.matmul(fc, _W['w_fc']) + _W['b_fc'])
    # fc = tf.nn.dropout(fc, dropout_rate)

    output = tf.matmul(fc, _W['w_out']) + _W['b_out']
    return output

# Construct model
output = model(x,weights)
# Softmax loss
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 100})

print('----------------------------------------------')
print('TRAINGING REFERENCE NET for LeNet-5')
print('----------------------------------------------')
################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
################### FOR REFERENCE NET #########################################
# Total minibatches
total_minibatches = 400
# number of minibatches in data
num_minibatches_data = data.train.images.shape[0] // minibatch

train_loss_steps = np.zeros(total_minibatches)
train_accuracy_steps = np.zeros(total_minibatches)
test_loss_steps = np.zeros(total_minibatches)
test_accuracy_steps = np.zeros(total_minibatches)

################### TO SAVE MODEL ##################
# model_file_name = 'reference_model_lenet_5.ckpt'
# model_file_path = './model_lenet_5/' + model_file_name 
saver = tf.train.Saver()
init = tf.global_variables_initializer()

############################## L-BFGS #########################################
with tf.Session() as sess:
	sess.run(init)
	for k in range(total_minibatches):
		feed_dict = {}
		index_minibatch = k % num_minibatches_data
		epoch = k // num_minibatches_data		
		# shuffle data at the begining of each epoch
		if index_minibatch == 0:
		 	X_train, y_train = shuffle_data(data)
 		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		feed_dict = {x: X_batch,
					 y: y_batch}

		optimizer.minimize(sess,feed_dict=feed_dict)
		
		############### LOSS AND ACCURACY EVALUATION ##########################
		train_loss, train_accuracy = \
				sess.run([loss, accuracy], feed_dict = {x: X_batch, 
													    y: y_batch} )
		train_loss_steps[k] = train_loss
		train_accuracy_steps[k] = train_accuracy

		val_loss, val_accuracy = \
		sess.run([loss, accuracy], feed_dict = {x: data.validation.images, 
												y: data.validation.labels} )
		
		test_loss_steps[k] = val_loss
		test_accuracy_steps[k] = val_accuracy

		print('step: {}, train loss: {}, train acuracy: {}' \
			.format(k, train_loss, train_accuracy) )
		print('step: {}, val loss: {}, val acuracy: {}' \
			.format(k, val_loss, val_accuracy) )

# save the results
# result_path = './results_LBFGS' + '_m_' + str(m) + '_minibatch_' + str(minibatch)
# np.savez(result_path,train_loss_steps=train_loss_steps,
# 					train_accuracy_steps=train_accuracy_steps,
# 					test_loss_steps = test_loss_steps,
# 					test_accuracy_steps=test_accuracy_steps)
