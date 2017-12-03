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
m = 20

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
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    # 'conv1': tf.Variable(tf.random_normal([F1, F1, D1, K1])),
    # 5x5 conv, 20 inputs, 50 outputs 
    #'conv2': tf.Variable(tf.random_normal([F3, F3, D3, K3])),
    'w_conv2': tf.get_variable('w_conv2', shape=[F3, F3, D3, K3],
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    # fully connected, 800 inputs, 500 outputs
    #'fc': tf.Variable(tf.random_normal([n_in_fc, n_hidden])),
    'w_fc': tf.get_variable('w_fc', shape=[n_in_fc, n_hidden],
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    # 500 inputs, 10 outputs (class prediction)
    #'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    'w_out': tf.get_variable('w_out', shape=[n_hidden, n_classes],
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    'b_conv1': tf.get_variable('b_conv1', shape=[K1],
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			# initializer=tf.zeros_initializer()),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    'b_conv2': tf.get_variable('b_conv2', shape=[K3],
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.zeros_initializer()),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc': tf.get_variable('b_fc', shape=[n_hidden],
           			initializer=tf.random_normal_initializer(stddev = 0.01)),
           			#initializer=tf.zeros_initializer()),
           			#initializer=tf.contrib.layers.xavier_initializer()),
    'b_out': tf.get_variable('b_out', shape=[n_classes],
           			initializer=tf.random_normal_initializer(stddev = 0.01))
           			# initializer=tf.zeros_initializer()) 
           			#initializer=tf.contrib.layers.xavier_initializer())
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
		S[i][layer] = np.random.rand(n_W[layer]).reshape(dim_W[layer])
		Y[i][layer] = np.random.rand(n_W[layer]).reshape(dim_W[layer])

def enqueue(mp,new_s_val,new_y_val):
	for layer, _ in weights.items():
		i = str(mp)
		S[i][layer] = new_s_val[layer]
		Y[i][layer] = new_y_val[layer]

def dequeue():
	for k in range(m-1):
		i = str(k)
		j = str(k+1)
		for layer, _ in weights.items():
			S[i][layer] = S[j][layer]
			Y[i][layer] = Y[j][layer]

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
alpha_tf = tf.placeholder("float",shape=[])
p_tf = {}
for layer, _ in weights.items():
	p_tf[layer] = tf.placeholder("float", shape=weights[layer].get_shape())

for layer, _ in weights.items():
	aux_w[layer] = tf.add(weights[layer], tf.multiply(alpha_tf,p_tf[layer]))

aux_output = model(x,aux_w)
aux_loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = aux_output))
aux_grad_w = {}
for layer, _ in weights.items():
	aux_grad_w[layer] = tf.gradients(xs=aux_w[layer], ys=aux_loss)

update_w = {}
update_w_placeholder = {}
for layer, _ in weights.items():
	update_w_placeholder[layer] = tf.placeholder(dtype="float",
										shape=weights[layer].get_shape())
for layer, _ in weights.items():
	update_w[layer] = weights[layer].assign(update_w_placeholder[layer])
###############################################################################
######################## LBFGS GRAPH ##########################################
###############################################################################
# q_tf = {}
# for layer, _ in weights.items():
# 	q_tf[layer] = tf.placeholder("float",weights[layer].get_shape())

# q = {}
# for layer, _ in weights.items():
# 	name = layer + 'q'
# 	q[layer] = tf.get_variable(shape=weights[layer].get_shape(),name=name)

# init_q = {}
# for layer, _ in weights.items():
# 	init_q[layer] = q[layer].assign(q_tf[layer])

# rho = {}
# sTq = {}
# yTr = {}
# yTs = {}
# alpha = {}
# beta = tf.Variable(initial_value=[0.0])
# gamma = tf.Variable(initial_value=[0.0])
# sTy = tf.Variable(initial_value=[0.0])
# # initialization
# for k in range(m-1,-1,-1):
# 	i = str(k)
# 	rho[i] = tf.Variable(initial_value=[1.0])
# 	sTq[i] = tf.Variable(initial_value=[0.0])
# 	yTr[i] = tf.Variable(initial_value=[0.0])
# 	yTs[i] = tf.Variable(initial_value=[0.0])
# 	alpha[i] = tf.Variable(initial_value=[0.0])

# zero_tf = tf.placeholder("float",shape=[None])
# one_tf = tf.placeholder("float",shape=[None])
# init_loop_var = {}
# init_loop_var[beta] = beta.assign(zero_tf)
# init_loop_var[gamma] = gamma.assign(zero_tf)
# init_loop_var[sTy] = sTy.assign(zero_tf)
# for k in range(m-1,-1,-1):
# 	i = str(k)
# 	init_loop_var[rho[i]] = rho[i].assign(zero_tf)
# 	init_loop_var[sTq[i]] = sTq[i].assign(zero_tf)
# 	init_loop_var[yTr[i]] = yTr[i].assign(zero_tf)
# 	init_loop_var[yTs[i]] = yTs[i].assign(zero_tf)
# 	init_loop_var[alpha[i]] = alpha[i].assign(zero_tf)

# def search_direction(mp=0):	
# 	for k in range(mp-1,-1,-1):
# 		i = str(k)
# 		for layer, _ in weights.items():
# 			yTs[i] = tf.add(yTs[i], \
# 				tf.reduce_sum( tf.multiply(S[i][layer],Y[i][layer] )))#))

# 		for layer, _ in weights.items():		
# 			rho[i] = tf.reciprocal(yTs[i])

# 		for layer, _ in weights.items():
# 			sTq[i] = tf.add( sTq[i] , \
# 				tf.reduce_sum(tf.multiply(S[i][layer],q[layer] )))
# 		alpha[i] = tf.multiply(rho[i],sTq[i])
		
# 		for layer, _ in weights.items():
# 			q[layer] = tf.subtract(q[layer], tf.multiply(alpha[i],Y[i][layer]))

# 	r = {}
# 	for layer, _ in weights.items():
# 		r[layer] = tf.identity(q[layer])
	
# 	global sTy
# 	if (mp >=1 ):
# 		for layer, _ in weights.items(): 
# 			sTy = tf.add(sTy , tf.reduce_sum(tf.multiply( 
# 												S[str(mp-1)][layer], 
# 												Y[str(mp-1)][layer])))
# 		gamma = tf.div(sTy,rho[str(mp-1)])
# 		for layer,_ in weights.items():
# 			r[layer] = tf.multiply(gamma , q[layer])

# 	for k in range(mp):
# 	 	i = str(k)
# 	 	for layer, _ in weights.items():
# 	 		yTr[i] = tf.add(yTr[i],tf.reduce_sum(tf.multiply(Y[i][layer],
# 	 														 r[layer] )))
# 	 	beta = tf.multiply(rho[i], yTr[i])

# 	 	for layer, _ in weights.items():
# 	 		r[layer] = tf.add(r[layer],tf.multiply(tf.subtract(alpha[i],beta),
# 	 															   S[i][layer]))
# 	p = {}
# 	for layer, _ in weights.items():
# 		p[layer] = tf.negative(r[layer])
# 	return p		
def search_direction(mp,old_grad_w):	
	q = old_grad_w
	yTs = {}
	rho = {}
	sTq = {}
	alpha = {}
	yTr = {}
	eps = np.finfo(float).eps
	for k in range(mp):
		i = str(k)
		yTs[i] = 0
		rho[i] = 0
		sTq[i] = 0
		alpha[i] = 0
		yTr[i] = 0

	for k in range(mp-1,-1,-1):
		i = str(k)
		for layer, _ in weights.items():
			yTs[i] = yTs[i] + np.dot(S[i][layer].flatten(),
									 Y[i][layer].flatten())		
		
		rho[i] = 1 / ( yTs[i] + eps) 

		for layer, _ in weights.items():
			sTq[i] = sTq[i] + np.dot(S[i][layer].flatten(),q[layer].flatten())
		alpha[i] = rho[i] * sTq[i]
		
		for layer, _ in weights.items():
			q[layer] = q[layer] - alpha[i] * Y[i][layer]

	r = q	
	
	if (mp >=1):
		sTy = 0
		yTy = 0
		for layer, _ in weights.items(): 
			sTy = sTy + np.dot( S[str(mp-1)][layer].flatten(),
								Y[str(mp-1)][layer].flatten())
			yTy = yTy + np.dot( Y[str(mp-1)][layer].flatten(),
								Y[str(mp-1)][layer].flatten())
		gamma = sTy / yTy
		for layer,_ in weights.items():
			r[layer] = gamma * q[layer]

	for k in range(mp):
	 	i = str(k)
	 	for layer, _ in weights.items():
	 		yTr[i] = yTr[i] + np.dot(Y[i][layer].flatten(), r[layer].flatten())

	 	beta = rho[i] * yTr[i]

	 	for layer, _ in weights.items():
	 		r[layer] = r[layer] + S[i][layer] * ( alpha[i] - beta )
	p = {}
	for layer, _ in weights.items():
		p[layer] = -1 * r[layer]
	return p		

################################################################################
################################################################################
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
total_minibatches = 400
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
 		# mini batch 
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

		# compute the subsampled gradient for minibatch of data
		old_grad_w_list = sess.run(grad_w, feed_dict=feed_dict)
		old_grad_w = {}
		for layer, _ in weights.items():
			old_grad_w[layer] = old_grad_w_list[layer][0]

		p_val = search_direction(mp,old_grad_w)
		# alpha_step, old_f, old_w, new_f, new_w, old_grad_w, new_grad_w = \
		# 			Wolfe_conditions(sess,feed_dict,grad_val,p_val)
		########################################################################
		############## FINDING ALPHA TO SATISFY ################################
		############## WOLFE CONDITIONS ########################################
		########################################################################
		alpha_step_vec = np.linspace(1,0,20,dtype='float')
		c1 = 1E-4
		c2 = 0.9
		for alpha_step in alpha_step_vec:
			feed_dict.update({alpha_tf: alpha_step})
			for layer, _ in weights.items():
				feed_dict.update({ p_tf[layer]: p_val[layer] })
			old_f = sess.run(loss, feed_dict=feed_dict)
			new_f = sess.run(aux_loss, feed_dict=feed_dict)
			old_w = sess.run(weights)
			new_w = sess.run(aux_w, feed_dict=feed_dict)
			gradTp = 0
			
			for layer, _ in weights.items():
				gradTp = gradTp + np.dot(old_grad_w[layer].flatten(),
										 p_val[layer].flatten())
			rhs = c1 * alpha_step * gradTp

			if new_f <= (old_f + rhs):
				Wolfe_cond_1 = True
			else:
				Wolfe_cond_1 = False

			new_grad_w = sess.run(aux_w,feed_dict=feed_dict)

			new_grad_wTp = 0
			for layer, _ in weights.items():
				new_grad_wTp = new_grad_wTp + np.dot(new_grad_w[layer].flatten(),
													 p_val[layer].flatten())
			if new_grad_wTp >= c2 * gradTp:
				Wolfe_cond_2 = True
			else:
				Wolfe_cond_2 = False

			if Wolfe_cond_1 and Wolfe_cond_2:
				break
		########################################################################
		############## UPDATE Storage S and Y ##################################
		########################################################################		
		new_s_val = {}
		new_y_val = {}
		for layer, _ in weights.items():
			new_s_val[layer] = new_w[layer] - old_w[layer]
			new_y_val[layer] = new_grad_w[layer] - old_grad_w[layer]
		
		if k < m:
			enqueue(k,new_s_val,new_y_val)
		else:
			dequeue()
			enqueue(m-1,new_s_val,new_y_val)
		########################################################################
		############## UPDATE Weights ##########################################
		########################################################################		
		feed_dict_w = {}
		for layer, _ in weights.items():
			feed_dict_w.update({update_w_placeholder[layer]: new_w[layer]})
		sess.run(update_w,feed_dict=feed_dict_w)

 		############### LOSS AND ACCURACY EVALUATION ##########################
		if k % 10 == 0:
			train_loss, train_accuracy = \
					sess.run([loss, accuracy], feed_dict = {x: X_batch, 
														    y: y_batch} )
			# train_loss_ref[epoch] = train_loss
			# train_error_ref[epoch] = 1 - train_accuracy

			val_loss, val_accuracy = \
			sess.run([loss, accuracy], feed_dict = {x: data.validation.images, 
													y: data.validation.labels} )
			# val_loss_ref[epoch] = val_loss
			# val_error_ref[epoch] = 1 - val_accuracy

			# test_loss, test_accuracy = \
			# sess.run([loss, accuracy], feed_dict = {x: data.test.images, 
			# 										y: data.test.labels} )
			# test_loss_ref[epoch] = test_loss
			# test_error_ref[epoch] = 1 - test_accuracy

			print('step: {}, train loss: {}, train acuracy: {}' \
				.format(epoch, train_loss, train_accuracy) )
			print('step: {}, val loss: {}, val acuracy: {}' \
				.format(epoch, val_loss, val_accuracy) )
			# print('step: {}, test loss: {}, test acuracy: {}' \
			# 	.format(epoch, test_loss, test_accuracy) )

		
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
# sTy_np = {}
# for key, _ in S_val.items():
# 	sTy_np[key] = 0
