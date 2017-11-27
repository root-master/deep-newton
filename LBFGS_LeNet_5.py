import tensorflow as tf

tf.reset_default_graph()

import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)

import numpy as np
import sys
import dill
import collections

import pickle

import pandas as pd

from sklearn.cluster import KMeans
from numpy import linalg as LA

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
m = 5

#mp = 4

# number of weights and bias in each layer
n_W = {}
n_b = {}

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
n_W['conv1'] = F1 * F1 * D1 * K1
n_b['conv1'] = K1 

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
n_W['conv2'] = F3 * F3 * D3 * K3
n_b['conv2'] = K3 

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
n_W['fc'] = n_in_fc * n_hidden
n_b['fc'] = n_hidden

# Layer 6 -- output
n_in_out = n_hidden
n_W['out'] = n_hidden * n_classes
n_b['out'] = n_classes

for key, value in n_W.items():
	n_W[key] = int(value)

for key, value in n_b.items():
	n_b[key] = int(value)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# learning_rate = tf.placeholder("float")
# momentum_tf = tf.placeholder("float")

# weights of LeNet-5 CNN -- tf tensors
# Calling everything weights from now on 

weights = {
    # 5 x 5 convolution, 1 input image, 20 outputs
    'w_conv1': tf.get_variable('w_conv1', shape=[F1, F1, D1, K1],
           			initializer=tf.contrib.layers.xavier_initializer()),
    # 'conv1': tf.Variable(tf.random_normal([F1, F1, D1, K1])),
    # 5x5 conv, 20 inputs, 50 outputs 
    #'conv2': tf.Variable(tf.random_normal([F3, F3, D3, K3])),
    'w_conv2': tf.get_variable('w_conv2', shape=[F3, F3, D3, K3],
           			initializer=tf.contrib.layers.xavier_initializer()),
    # fully connected, 800 inputs, 500 outputs
    #'fc': tf.Variable(tf.random_normal([n_in_fc, n_hidden])),
    'w_fc': tf.get_variable('w_fc', shape=[n_in_fc, n_hidden],
           			initializer=tf.contrib.layers.xavier_initializer()),
    # 500 inputs, 10 outputs (class prediction)
    #'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    'w_out': tf.get_variable('w_out', shape=[n_hidden, n_classes],
           			initializer=tf.contrib.layers.xavier_initializer()),
    'b_conv1': tf.get_variable('b_conv1', shape=[K1],
           			initializer=tf.zeros_initializer()),
    'b_conv2': tf.get_variable('b_conv2', shape=[K3],
           			initializer=tf.zeros_initializer()),
    'b_fc': tf.get_variable('b_fc', shape=[n_hidden],
           			initializer=tf.zeros_initializer()),
    'b_out': tf.get_variable('b_out', shape=[n_classes],
           			initializer=tf.zeros_initializer()) 
}

S = {}
Y = {}

for k in range(m):
	i = str(k)
	S[str(k)] = {}
	Y[str(k)] = {}

for k in range(m):
	i = str(k)
	for layer, _ in weights.items():
		name = layer + '_s_store_' + i
		S[i][layer] = tf.get_variable(	name=name,
										shape=weights[layer].get_shape(),
										initializer=tf.contrib.layers.xavier_initializer())
		name = layer + '_y_store_' + i
		Y[i][layer] = tf.get_variable(	name=name,
										shape=weights[layer].get_shape(),
										initializer=tf.contrib.layers.xavier_initializer())
new_s = {}
new_y = {}

for layer, _ in weights.items():
	name = layer + '_new_s'
	new_s[layer] = tf.get_variable(	name=name,
									shape=weights[layer].get_shape(),
									initializer=tf.contrib.layers.xavier_initializer())

for layer, _ in weights.items():
	name = layer + '_new_y'
	new_y[layer] = tf.get_variable(	name=name,
									shape=weights[layer].get_shape(),
									initializer=tf.contrib.layers.xavier_initializer())


def enqueue(queue=None, k=0, new_s=None, new_y=None):
	if queue is None:
		pass
	#if k < len(queue.keys()):



def dequeue():
	pass

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


###############################################################################
######################## TF GRADINETS #########################################
###############################################################################
grad_w = {}
for layer, _ in weights.items():
	grad_w[layer] = tf.gradients(xs=weights[layer], ys=loss)

###############################################################################
######################## TF Auxilary variables ################################
###############################################################################
aux_w = {}
for layer, _ in weights.items():
	aux_w[layer] = tf.identity(weights[layer])

aux_loss = model(x,aux_w)
aux_grad_w = {}
for layer, _ in weights.items():
	aux_grad_w[layer] = tf.gradients(xs=aux_w[layer], ys=aux_loss)

###############################################################################
######################## LBFGS GRAPH ##########################################
###############################################################################

###############################################################################
######################## UPDATE GRADS #########################################
###############################################################################
# step size
# alpha_step = tf.placeholder("float",shape=[])
# search direction
# p = {}
# p = {}
# for layer, _ in weights.items():
# 	name = 'p_' + layer
# 	p[layer] = tf.get_variable(name=name, 
# 							  shape=weights[layer].get_shape())

#with tf.name_scope('L-BFGS-two-loop-recurssion'):

def search_direction(mp=0):
	rho = {}
	sTq = {}
	yTr = {}
	alpha = {}
	q = {}
	for layer, _ in weights.items():
		q[layer] = tf.squeeze( tf.identity(grad_w[layer]),axis=0)
	#mp = tf.placeholder("int")
	
	# initialization
	for k in range(m-1,-1,-1):
		i = str(k)
		rho[i] = tf.Variable(initial_value=[0.1])
		sTq[i] = tf.Variable(initial_value=[0.1])
		yTr[i] = tf.Variable(initial_value=[0.1])
		alpha[i] = tf.Variable(initial_value=[0.1])
		beta = tf.Variable(initial_value=[0.1])

	gamma = tf.Variable(initial_value=[0.1])

	sTy = tf.Variable(initial_value=[1.0])

	for k in range(mp-1,-1,-1):
		i = str(k)
		for layer, _ in weights.items():
			rho[i].assign(rho[i] + 1 / ( tf.reduce_sum(
							tf.multiply(S[i][layer],Y[i][layer] ))))
		for layer, _ in weights.items():
			sTq[i].assign(sTq[i] + \
				( tf.reduce_sum(tf.multiply(S[i][layer],q[layer] ))))
		alpha[i] = tf.multiply(rho[i],sTq[i])
		
		for layer, _ in weights.items():
			q[layer] = q[layer] - tf.multiply(alpha[i],Y[i][layer])

	r = {}
	for layer, _ in weights.items():
		r[layer] = tf.identity(q[layer])
	if (mp >=1 ):
		for layer, _ in weights.items(): 
			sTy.assign(tf.add(sTy , tf.reduce_sum(tf.multiply( 
												S[str(mp-1)][layer], 
												Y[str(mp-1)][layer]))))
		gamma = sTy / rho[str(mp-1)]
		for layer,_ in weights.items():
			r[layer] = tf.multiply(gamma , q[layer])

	for k in range(mp):
		i = str(k)
		for layer, _ in weights.items():
			yTr[i].assign(yTr[i] + \
				( tf.reduce_sum(tf.multiply(Y[i][layer],r[layer] ))))
		beta = tf.multiply( rho[i], yTr[i] )

		for layer, _ in weights.items():
			r[layer] = tf.add(r[layer],
									tf.multiply(
										tf.subtract( alpha[i],beta ),
										S[i][layer]))
	return r		


#with tf.name_scope('L-BFGS-update'):
# One iteration of L-BFGS
def body():
	return pass

def cond(alpha_tf):
	Wolfe = tf.cond(tf.logical_and(cond_1,cond_2),lambda: tf.constant(True))
	return pass

def Wolfe_conditions(p,old_grad_w):
	alpha_step_vec = np.linspace(1,0,20,dtype='float')
	c1 = 1E-4
	c2 = 0.9

	Wolfe = tf.Variable(False)
	for alpha_step in alpha_step_vec:
		new_w = {}
		alpha_tf = tf.Variable(initial_value=alpha_step,dtype='float')
		for layer, _ in weights.items():
			new_w[layer] = weights[layer] + tf.scalar_mul(  alpha_tf,
															p[layer])		
		new_output = model(x, new_w)
		new_loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels = y, 
													logits = new_output))

		rhs_1= tf.Variable([0.0])
		for layer, _ in weights.items():
			rhs_1 + tf.multiply(c1*alpha_tf,
							  tf.reduce_sum(tf.multiply( p[layer], 
									                     old_grad_w[layer])))
		rhs_1 = rhs_1 + output
		lhs_1 = new_loss

		cond_1 = lambda: tf.greater_equal(rhs_1,lhs_1)
			# ops to update weights and biases given search direction and step size 
		
		for layer, _ in weights.items():
			aux_w[layer] = tf.identity(new_w[layer])

		new_grad_w = {}
		for layer, _ in weights.items():
			new_grad_w[layer] = tf.squeeze(tf.identity(aux_grad_w[layer]),axis=0)

		# new_grad_w = {}		
		# for layer, _ in weights.items():
		# 	new_grad_w[layer] = tf.squeeze(new_grad_w_op[layer])

		lhs_2 = tf.Variable([0.0])
		for layer, _ in weights.items():
			lhs_2 + tf.reduce_sum(tf.multiply(p[layer],new_grad_w[layer]))
		rhs_2 = tf.Variable([0.0])
		for layer, _ in weights.items():
			rhs_1 + tf.multiply(c2,tf.reduce_sum(tf.multiply(p[layer], 
									                         old_grad_w[layer])))
		cond_2 = lambda: tf.less_equal(rhs_2,lhs_2)
		Wolfe = tf.cond(tf.logical_and(cond_1,cond_2),lambda: tf.constant(True))
		
	
	return alpha_tf, new_w, new_grad_w

def LBFGS_update(mp=0):
	# current w
	old_w = {}
	for layer, _ in weights.items():
		old_w[layer] = tf.identity(weights[layer])

	old_grad_w = {}
	for layer, _ in weights.items():
		old_grad_w[layer] = tf.squeeze( tf.identity(grad_w[layer]),axis=0)
		
	
	# ops to update weights and biases given search direction and step size 
	new_w = {}	
	p = search_direction(mp)
	
	alpha_tf, new_w, new_grad_w = Wolfe_conditions(p,old_grad_w)
	
	for layer, _ in weights.items():
		new_w[layer] = weights[layer] + tf.multiply(alpha_step , p[layer])

	#update_cond = tf.placeholder('float')
	for layer, _ in weights.items():
		weights[layer] = tf.identity(new_w)

	new_grad_w = {}
	for layer, _ in weights.items():
		new_grad_w[layer] = tf.squeeze( tf.identity(grad_w[layer]),axis=0)

	for layer, _ in weights.items():
		new_s[layer] = new_w[layer] - old_w[layer]

	for layer, _ in weights.items():
		new_y[layer] = new_grad_w[layer] - old_grad_w[layer]


	return old_w, new_w, old_grad_w, new_grad_w, new_s, new_y

saver = tf.train.Saver()
init = tf.global_variables_initializer()

###############################################################################
######## training data and neural net architecture with weights w #############
###############################################################################
print('----------------------------------------------')
print('TRAINGING REFERENCE NET for LeNet-5')
print('----------------------------------------------')
################### TO SAVE TRAINING AND TEST LOSS AND ERROR ##################
################### FOR REFERENCE NET #########################################
# Batch size
minibatch = 512
# Total minibatches
total_minibatches = 30
# number of minibatches in data
num_minibatches_data = data.train.images.shape[0] // minibatch

num_epoch_ref = total_minibatches // num_minibatches_data
epoch_ref_vec = np.array(range(num_epoch_ref+1)) 
train_loss_ref = np.zeros(num_epoch_ref+1)
train_error_ref = np.zeros(num_epoch_ref+1)
val_loss_ref = np.zeros(num_epoch_ref+1)
val_error_ref = np.zeros(num_epoch_ref+1)
test_loss_ref = np.zeros(num_epoch_ref+1)
test_error_ref = np.zeros(num_epoch_ref+1)

################### TO SAVE MODEL ##################
# model_file_name = 'reference_model_lenet_5.ckpt'
# model_file_path = './model_lenet_5/' + model_file_name 

############################## L-BFGS #########################################
with tf.Session() as sess:
	sess.run(init)
	for k in range(total_minibatches):
		index_minibatch = k % num_minibatches_data
		epoch = k // num_minibatches_data		
		# shuffle data at the begining of each epoch
		if index_minibatch == 0:
		 	X_train, y_train = shuffle_data(data)
# 		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]

		if k < m:
			mp = k
		else:
			mp = m
		feed_dict = {x: X_batch,
					y: y_batch}
		old_w_v, new_w_v, old_grad_w_v, new_grad_w_v, new_s_v, new_y_v = sess.run(LBFGS_update(mp),feed_dict=feed_dict)
		
# 		############### LOSS AND ACCURACY EVALUATION ##########################
		if index_minibatch == 0:
			train_loss, train_accuracy = \
					sess.run([loss, accuracy], feed_dict = {x: X_batch, 
														    y: y_batch} )
			train_loss_ref[epoch] = train_loss
			train_error_ref[epoch] = 1 - train_accuracy

			val_loss, val_accuracy = \
			sess.run([loss, accuracy], feed_dict = {x: data.validation.images, 
													y: data.validation.labels} )
			val_loss_ref[epoch] = val_loss
			val_error_ref[epoch] = 1 - val_accuracy

			test_loss, test_accuracy = \
			sess.run([loss, accuracy], feed_dict = {x: data.test.images, 
													y: data.test.labels} )
			test_loss_ref[epoch] = test_loss
			test_error_ref[epoch] = 1 - test_accuracy

			print('step: {}, train loss: {}, train acuracy: {}' \
				.format(i, train_loss, train_accuracy) )
			print('step: {}, val loss: {}, val acuracy: {}' \
				.format(i, val_loss, val_accuracy) )
			print('step: {}, test loss: {}, test acuracy: {}' \
				.format(i, test_loss, test_accuracy) )

		
# 	save_path = saver.save(sess, model_file_path)
# 	# reference weight and bias
# 	w_bar = sess.run(weights)
# 	bias_bar = sess.run(biases)


# df_ref = pd.DataFrame({	'train_loss_ref' : train_loss_ref,
# 						'train_error_ref': train_error_ref,
# 						'val_loss_ref': val_loss_ref,
# 						'val_error_ref': val_error_ref,
# 						'test_loss_ref': test_loss_ref,
# 						'test_error_ref': test_error_ref})


# file_pickle = './results_lenet_5/df_ref_lenet_5_pickle.pkl'
# with open(file_pickle,'wb') as f:
# 	df_ref.to_pickle(f)

# weights_pickle = './results_lenet_5/weights_biases_lenet_5_ref_pickle.pkl'

# with open(weights_pickle,'wb') as f:
# 	pickle.dump(w_bar,f,protocol=pickle.HIGHEST_PROTOCOL)
# 	pickle.dump(bias_bar,f,protocol=pickle.HIGHEST_PROTOCOL)

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
#     saver.restore(sess, "/tmp/model.ckpt")
# reference weight and bias
	# w_bar = sess.run(weights)
	# bias_bar = sess.run(biases)