import numpy as np
import tensorflow as tf

def conv2d(x, n_output,
		   k_h=5, k_w=5, d_h=2, d_w=2,
		   padding='SAME', activation=tf.nn.sigmoid, name='conv2d', reuse=None):
	"""Helper for creating a 2d convolution operation.
	Parameters
	----------
	x : tf.Tensor
		Input tensor to convolve.
	n_output : int
		Number of filters.
	k_h : int, optional
		Kernel height
	k_w : int, optional
		Kernel width
	d_h : int, optional
		Height stride
	d_w : int, optional
		Width stride
	padding : str, optional
		Padding type: "SAME" or "VALID"
	activation : fn
		Activation function
	name : str, optional
		Variable scope
	Returns
	-------
	op : tf.Tensor
		Output of convolution
	"""

	print("conv2d layer", x.get_shape())

	with tf.variable_scope(name or 'mir3_conv2d', reuse=reuse):
		W = tf.get_variable(
			name='W',
			shape=[k_h, k_w, x.get_shape()[-1], n_output],
			initializer=tf.contrib.layers.xavier_initializer_conv2d())

		#print W.get_shape()

		conv = tf.nn.conv2d(
			name='conv',
			input=x,
			filter=W,
			strides=[1, d_h, d_w, 1],
			padding=padding)

		b = tf.get_variable(
			name='b',
			shape=[n_output],
			initializer=tf.constant_initializer(0.0))

		h = tf.nn.bias_add(
			name='h',
			value=conv,
			bias=b)

		if activation:
			h = activation(h)

	return h, W

def linear(x, n_output, name=None, activation=tf.nn.sigmoid, reuse=None):
	"""Fully connected layer.
	Parameters
	----------
	x : tf.Tensor
		Input tensor to connect
	n_output : int
		Number of output neurons
	name : None, optional
		Scope to apply
	Returns
	-------
	h, W : tf.Tensor, tf.Tensor
		Output of fully connected layer and the weight matrix
	"""
	if len(x.get_shape()) != 2:
		x = flatten(x, reuse=reuse)

	n_input = x.get_shape().as_list()[1]

	print("Linear layer input=", n_input, "output=", n_output)

	with tf.variable_scope(name or "mir3_fc", reuse=reuse):
		W = tf.get_variable(
			name='W',
			shape=[n_input, n_output],
			dtype=tf.float32,
			initializer=tf.contrib.layers.xavier_initializer())

		b = tf.get_variable(
			name='b',
			shape=[n_output],
			dtype=tf.float32,
			initializer=tf.constant_initializer(0.0))

		preactivate = tf.nn.bias_add(
			name='h',
			value=tf.matmul(x, W),
			bias=b)

		tf.summary.histogram('pre_activations', preactivate)

		activations = preactivate

		if activation:
			activations = activation(preactivate)

		tf.summary.histogram('activations', activations)    

		return activations, W

def build_model(n_bands, n_samples, n_classes, learning_rate, model_name='medley-1'):

	activation_fn = tf.nn.relu #sigmoid
	conv_layers = [64, 128, 256, 256]
	kernel = [3,3]

	X = tf.placeholder(tf.float32, shape=[None, n_bands, n_samples, 1], name='X')
	Y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
	keep_prob = tf.placeholder(tf.float32)

	prev_output = X
	conv_index = 0
	for conv_size in conv_layers:

		h = conv2d(prev_output, conv_size, k_h=kernel[0], k_w=kernel[1], d_h=1, d_w=1, name='conv2d_layer_' + str(conv_index), activation=activation_fn)[0]
		h = conv2d(h, conv_size, k_h=kernel[0], k_w=kernel[1], d_h=1, d_w=1, name='conv2d_layer_' + str(conv_index+1), activation=activation_fn)[0]
		#h = conv2d(h, conv_size, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2d_layer_' + str(conv_index+2), activation=activation_fn)[0]
		h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
		with tf.name_scope('dropout'):
			h = tf.nn.dropout(h, keep_prob)
		conv_index += 2	
		prev_output = h

	h = prev_output
	
	# TODO: add dropout 0.25

	# flatten
	shape = h.get_shape().as_list()
	print('shape before flatten', shape)
	h = tf.reshape(h, [-1, shape[1] * shape[2] * shape[3]])

	# fc
	#h = linear(h, 2*2048, name='fc_layer_1', activation=activation_fn)[0]
	h = linear(h, 2048, name='fc_layer_1', activation=activation_fn)[0]
	h = linear(h, 2048, name='fc_layer_2', activation=activation_fn)[0]
	h = linear(h, 2048, name='fc_layer_3', activation=activation_fn)[0]
	h, w4 = linear(h, 2048, name='fc_layer_4', activation=activation_fn)
	#h = linear(h, 1024, name='fc_layer_2', activation=activation_fn)[0]
	#h = linear(h, 512, name='fc_layer_3', activation=activation_fn)[0]
	#h = linear(h, 256, name='fc_layer_4', activation=activation_fn)[0]
	#h = linear(h, 128, name='fc_layer_5', activation=activation_fn)[0]
	#h = linear(h, 64, name='fc_layer_6', activation=activation_fn)[0]
	#h = linear(h, 32, name='fc_layer_7', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_2', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_3', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_4', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_5', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_6', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_7', activation=activation_fn)[0]
	#h = linear(h, 2048, name='fc_layer_8', activation=activation_fn)[0]

	# logits
	logits = linear(h, n_classes, name='fc_layer_logits', activation=activation_fn)[0]

	# softmax, resulting shape=[-1, n_classes]
	Y_pred = tf.nn.softmax(logits, name='softmax_layer')

	diff =  tf.nn.softmax_cross_entropy_with_logits(logits, Y)
	cross_entropy = tf.reduce_mean(diff, name='cross_entropy')

	
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	with tf.name_scope(model_name):
		tf.summary.scalar('cross_entropy', cross_entropy)
		tf.summary.scalar('train_accuracy', accuracy)
	summary = tf.summary.merge_all()
	return X, Y, keep_prob, Y_pred, cross_entropy, optimizer, summary, accuracy, w4




