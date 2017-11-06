import tensorflow as tf

import input_MNIST_data
from input_MNIST_data import shuffle_data
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)

import numpy as np
import sys

from sklearn.cluster import KMeans
from numpy import linalg as LA

# input and output shape
n_input   = data.train.images.shape[1]  # here MNIST data input (28,28)
n_classes = data.train.labels.shape[1]  # here MNIST (0-9 digits)

# Network Parameters
n_hidden_1 = 300  # 1st layer num features
n_hidden_2 = 100  # 2nd layer num features


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
learning_rate = tf.placeholder("float")
momentum_tf = tf.placeholder("float")


def model(_X, _W, _bias):
    # Hidden layer with tanh activation
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _W['fc1']), _bias['fc1']))  
    # Hidden layer with tanh activation
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, _W['fc2']), _bias['fc2']))  
    # output without any activation
    output = tf.add(tf.matmul(layer_2, _W['out']) , _bias['out'])
    return output

def model_compression(_X, _wC_tf, _biasC_tf):
    # Hidden layer with tanh activation
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _wC_tf['fc1']), _biasC_tf['fc1']))  
    # Hidden layer with tanh activation
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, _wC_tf['fc2']), _biasC_tf['fc2']))  
    # output without any activation
    output_compression = tf.add(tf.matmul(layer_2, _wC_tf['out']) , _biasC_tf['out'])
    return output_compression


# Store layers weight & bias
W = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}

bias = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}

# Construct model
output = model(x, W, bias)

# Define loss and optimizer
# Softmax loss
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))

# L step tf
mu_tf = tf.placeholder("float")
wC_tf = {
    'fc1': tf.placeholder("float", [n_input, n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_1, n_hidden_2]),
    'out': tf.placeholder("float", [n_hidden_2, n_classes])
}
biasC_tf = {
	'fc1': tf.placeholder("float", [n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_2]),
    'out': tf.placeholder("float", [n_classes])
}

lamda_tf = {
	'fc1': tf.placeholder("float", [n_input, n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_1, n_hidden_2]),
    'out': tf.placeholder("float", [n_hidden_2, n_classes])
}

lamda_bias_tf = {
	'fc1': tf.placeholder("float", [n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_2]),
    'out': tf.placeholder("float", [n_classes])
}

w_init_placeholder = {
    'fc1': tf.placeholder("float", [n_input, n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_1, n_hidden_2]),
    'out': tf.placeholder("float", [n_hidden_2, n_classes])
}
bias_init_placeholder = {
	'fc1': tf.placeholder("float", [n_hidden_1]),
    'fc2': tf.placeholder("float", [n_hidden_2]),
    'out': tf.placeholder("float", [n_classes])
}

w_init = {
 	'fc1' : W['fc1'].assign(w_init_placeholder['fc1']),
 	'fc2' : W['fc2'].assign(w_init_placeholder['fc2']),
 	'out' : W['out'].assign(w_init_placeholder['out'])
}

bias_init = {
 	'fc1' : bias['fc1'].assign(bias_init_placeholder['fc1']),
 	'fc2' : bias['fc2'].assign(bias_init_placeholder['fc2']),
 	'out' : bias['out'].assign(bias_init_placeholder['out'])
}

norm_tf = tf.norm( W['fc1'] - wC_tf['fc1'] - lamda_tf['fc1'] / mu_tf ,ord='euclidean') \
	    + tf.norm( W['fc2'] - wC_tf['fc2'] - lamda_tf['fc2'] / mu_tf ,ord='euclidean') \
	    + tf.norm( W['out'] - wC_tf['out'] - lamda_tf['out'] / mu_tf,ord='euclidean') \
	    + tf.norm( bias['fc1'] - biasC_tf['fc1'] - lamda_bias_tf['fc1'] / mu_tf ,ord='euclidean') \
	    + tf.norm( bias['fc2'] - biasC_tf['fc2'] - lamda_bias_tf['fc2'] / mu_tf ,ord='euclidean') \
	    + tf.norm( bias['out'] - biasC_tf['out'] - lamda_bias_tf['out'] / mu_tf ,ord='euclidean')

regularizer = mu_tf / 2 * norm_tf

loss_L_step =  loss + regularizer 

# learning_rate = tf.placeholder("float")

# grad_w = {
#     'fc1': tf.gradients(loss, W['fc1']),
#     'fc2': tf.gradients(loss, W['fc2']),
#     'out': tf.gradients(loss, W['out'])
# }

# grad_bias = {
#     'fc1': tf.gradients(loss, bias['fc1']),
#     'fc2': tf.gradients(loss, bias['fc1']),
#     'out': tf.gradients(loss, bias['fc1'])
# }

# new_W = {
# 	'fc1' : W['fc1'].assign(W['fc1'] - learning_rate * grad_w['fc1']),
# 	'fc2' : W['fc2'].assign(W['fc2'] - learning_rate * grad_w['fc2']),
# 	'out' : W['out'].assign(W['out'] - learning_rate * grad_w['out'])
# }

# new_bias = {
# 	'fc1' : bias['fc1'].assign(bias['fc1'] - learning_rate * grad_bias['fc1']),
# 	'fc2' : bias['fc2'].assign(bias['fc2'] - learning_rate * grad_bias['fc2']),
# 	'out' : W['out'].assign(bias['out'] - learning_rate * grad_bias['out'])
# }



#Training the Reference model: 

# Batch size: 512
minibatch = 512
# Total minibatches
total_minibatches = 100000
# number of minibatches in data
num_minibatches_data = data.train.images.shape[0] // minibatch

# Learning rate
lr = 0.02
# Learning rate decay:  every 2000 minibatches
learning_rate_decay = 0.98
learning_rate_stay_fixed = 2000

# Optimizer: Nesterov accelerated gradient with momentum 0.95
# this is for training the reference net
momentum = 0.9

optimizer = tf.train.MomentumOptimizer(
	learning_rate = learning_rate,
	momentum = momentum_tf,
	use_locking=False,
	name='Momentum',
	use_nesterov=True)

GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2
# GATE_OP:
# For each Op, make sure all gradients are computed before
# they are used.  This prevents race conditions for Ops that generate gradients
# for multiple inputs where the gradients depend on the inputs.
train = optimizer.minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name='train',
    grad_loss=None)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# compression output
output_compression = model_compression(x, wC_tf, biasC_tf)
correct_prediction_compression = tf.equal(tf.argmax(output_compression, 1), tf.argmax(y, 1))
accuracy_compression = tf.reduce_mean(tf.cast(correct_prediction_compression, tf.float32))

saver = tf.train.Saver()

train_L_step = optimizer.minimize(
    loss_L_step,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name='train_L_step',
    grad_loss=None)

init = tf.global_variables_initializer()

###############################################################################
######## training data and neural net architecture with weights w #############
###############################################################################
with tf.Session() as sess:
	sess.run(init)
	for i in range(total_minibatches):
		index_minibatch = i % num_minibatches_data
		epoch = i // num_minibatches_data		
		# shuffle data at the begining of each epoch
		if index_minibatch == 0:
			X_train, y_train = shuffle_data(data)
		# adjust learning rate
		if i % learning_rate_stay_fixed == 0:
			j = i // learning_rate_stay_fixed
			lr = learning_rate_decay ** j
		# mini batch 
		start_index = index_minibatch     * minibatch
		end_index   = (index_minibatch+1) * minibatch
		X_batch = X_train[start_index:end_index]
		y_batch = y_train[start_index:end_index]
		
		if i % 100 == 0:
			train_accuracy = accuracy.eval(
				feed_dict={x: X_batch, 
						   y: y_batch})
			print('step {}, training accuracy {}' .format(i, train_accuracy))

		if index_minibatch == 0:
			print('epoch {} and test accuracy {}' .format(epoch, accuracy.eval(
				feed_dict={x: data.validation.images, 
						   y: data.validation.labels})))
		# train on batch
		feed_dict = {x: X_batch,
					 y: y_batch,
					 learning_rate: lr,
					 momentum_tf: momentum}
		train.run(feed_dict)
	
	save_path = saver.save(sess, "./model/reference_net_final.ckpt")
	# reference weight and bias
	w_bar = sess.run(W)
	bias_bar = sess.run(bias)

###############################################################################
################### learn codebook and assignments ############################
###############################################################################
# flatten the weights and concatenate bias for each layer
w = {}
for layer, weight_matrix in w_bar.items():
	wf = weight_matrix.flatten()
	wf = np.concatenate( (wf , bias_bar[layer]) , axis=0)
	w[layer] = wf.reshape(-1 , 1)

# codebook size
k = 16
# dictionary to save the kmeans output for each layer 
kmeans = {}
# codebook of each layer i.e. centers of kmeans
C = {}
# assignments i.e. labels of kmeans
Z = {}
# quantized reference net i.e. prediction of kmeans
wC = {}

# Kmeans
for layer, _ in w.items():
	kmeans[layer] = KMeans(n_clusters=k, random_state=0).fit(w[layer])
	C[layer] = kmeans[layer].cluster_centers_ 
	Z[layer] = kmeans[layer].labels_
	# quantize reference net
	wC[layer]= C[layer][Z[layer]]

###############################################################################
####################### reshape weights #######################################
wC_reshape = {}
biasC = {}
for layer, _ in w_bar.items():
	wC_reshape[layer] = wC[layer][0:w_bar[layer].size].reshape(w_bar[layer].shape)
	biasC[layer] = wC[layer][w_bar[layer].size:].reshape(-1)
###############################################################################
###############################################################################
# initilize lambda == python reserved lambda so let's use lamda
lamda = {}
lamda_bias = {}
for layer, _ in w_bar.items():
	lamda[layer] = np.zeros(w_bar[layer].shape)
	lamda_bias[layer] = np.zeros(bias_bar[layer].shape).reshape(-1)

###############################################################################
####################################### LC ####################################
momentum = 0.95
# mu parameters
mu_0 = 9.75e-5
a = 1.1
max_iter_each_L_step = 2000
LC_epoches = 30
random_w_init = 0 # 0: random init, 1 if init with reference net
with tf.Session() as sess: 
	###########################################################################
	######## Initilize weights and bias #######################################
	if random_w_init:
		# initilize weights and bias randomly
		sess.run(init)
	else:
		sess.run(init)
		# initilize weights and bias with reference net
		feed_dict = {
			w_init_placeholder['fc1']: w_bar['fc1'],
			w_init_placeholder['fc2']: w_bar['fc2'],
			w_init_placeholder['out']: w_bar['out'],
			bias_init_placeholder['fc1']: bias_bar['fc1'],
			bias_init_placeholder['fc2']: bias_bar['fc2'],
			bias_init_placeholder['out']: bias_bar['out']
		}
		sess.run([w_init,bias_init], feed_dict=feed_dict)
	for j in range(LC_epoches):
		print('L step {} : ' .format(j))
		# adjust mu
		mu = mu_0 * ( a ** j )
		# adjust learning rate
		lr = 0.1 * ( 0.99 ** j )
		#######################################################################
		######## L Step #######################################################
		#######################################################################	
		# variable.initialized_value() ?
		for i in range(max_iter_each_L_step):
			index_minibatch = i % num_minibatches_data
			epoch = i // num_minibatches_data		
			# shuffle data at the begining of each epoch
			if index_minibatch == 0:
				X_train, y_train = shuffle_data(data)
			# mini batch 
			start_index = index_minibatch     * minibatch
			end_index   = (index_minibatch+1) * minibatch
			X_batch = X_train[start_index:end_index]
			y_batch = y_train[start_index:end_index]
		
			if i % 100 == 0:
				train_accuracy = accuracy.eval(
					feed_dict = {x: X_batch, 
						   		 y: y_batch})
				print('step {}, training accuracy {}' .format(i, train_accuracy))

			###################################################################
			####################### training batch in L #######################
			# train on batch
			feed_dict = {x: X_batch,
						 y: y_batch,
						 learning_rate: lr,
						 momentum_tf: momentum,
						 mu_tf: mu,
						 wC_tf['fc1']: wC_reshape['fc1'],
						 wC_tf['fc2']: wC_reshape['fc2'],
						 wC_tf['out']: wC_reshape['out'],
						 biasC_tf['fc1']: biasC['fc1'],
						 biasC_tf['fc2']: biasC['fc2'],
						 biasC_tf['out']: biasC['out'],
						 lamda_tf['fc1']: lamda['fc1'],
						 lamda_tf['fc2']: lamda['fc2'],
						 lamda_tf['out']: lamda['out'],
						 lamda_bias_tf['fc1']: lamda_bias['fc1'],
						 lamda_bias_tf['fc2']: lamda_bias['fc2'],
						 lamda_bias_tf['out']: lamda_bias['out']}
			train_L_step.run(feed_dict)
			# reference weight and bias
			w_bar = sess.run(W)
			bias_bar = sess.run(bias)
		######################################################################
		####################### accuracy using w #############################
		print('epoch {} and test accuracy using w {}' .format(j, accuracy.eval(
			feed_dict={x: data.validation.images, 
					   y: data.validation.labels})))

		#######################################################################
		######## C Step #######################################################
		#######################################################################
		# flatten the weights and concatenate bias for each layer
		w = {}
		for layer, _ in w_bar.items():
			wf = w_bar[layer].flatten() - lamda[layer].flatten() / mu
			bf = bias_bar[layer] - lamda_bias[layer] / mu
			wf = np.concatenate( (wf , bf) , axis=0)
			w[layer] = wf.reshape(-1 , 1)

		# Kmeans
		for layer, _ in w.items():
			kmeans[layer] = KMeans(n_clusters=k, random_state=0).fit(w[layer])
			C[layer] = kmeans[layer].cluster_centers_ 
			Z[layer] = kmeans[layer].labels_
			# quantize reference net
			wC[layer]= C[layer][Z[layer]]
		######################################################################
		####################### reshape weights ##############################
		for layer, _ in w_bar.items():
			wC_reshape[layer] = wC[layer][0:w_bar[layer].size].reshape(w_bar[layer].shape)
			biasC[layer] = wC[layer][w_bar[layer].size:].reshape(-1)
		
		######################################################################
		####################### accuracy using wc ############################
		print('epoch {} and test accuracy using wc {}' 
						.format(j, accuracy_compression.eval(
							feed_dict={x: data.validation.images, 
									   y: data.validation.labels,
									   wC_tf['fc1']: wC_reshape['fc1'],
									   wC_tf['fc2']: wC_reshape['fc2'],
									   wC_tf['out']: wC_reshape['out'],
									   biasC_tf['fc1']: biasC['fc1'],
									   biasC_tf['fc2']: biasC['fc2'],
									   biasC_tf['out']: biasC['out']})))
		#######################################################################
		############################ update lambda ############################
		for layer, _ in w_bar.items():
			lamda[layer] = lamda[layer] - mu * (w_bar[layer] - wC_reshape[layer])
			lamda_bias[layer] = lamda_bias[layer] - mu * (bias_bar[layer] - biasC[layer])

		norm_compression = 0
		for layer, _ in w_bar.items():
			norm_compression = LA.norm(w[layer] - wC[layer])

		print('norm of compression: {} ' .format(norm_compression) )

		if norm_compression < 0.001:
			break

	save_path = saver.save(sess, "./model/compressed_net_final.ckpt")
















