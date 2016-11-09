import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def AE(input, is_training=True, alpha=1, beta=0.25):
	batch_size, items = [x.value for x in input.get_shape()]
	net = slim.dropout(input, 0.5, is_training=is_training)
	print "Net", net.get_shape()
	if batch_size:
		mask = tf.random_uniform([batch_size, items], maxval=2, dtype=tf.int32) # randomint 0 or 1
	else:
		mask = tf.ones_like(net) #all to reconstruction
	mask = tf.to_float(mask)
	imask = 1 - mask
	corrupted = tf.mul(net, mask)

	encoder = slim.fully_connected(corrupted, 500, tf.nn.tanh)
	decoder = slim.fully_connected(encoder, items, tf.nn.tanh)	
	
	denoising_loss = slim.losses.mean_squared_error(tf.mul(imask,decoder), tf.mul(imask,input))
	tf.scalar_summary('losses/Denoising', denoising_loss)
	reconstruction_loss = slim.losses.mean_squared_error(tf.mul(mask,decoder),tf.mul(mask,input))
	tf.scalar_summary('losses/Reconstruction', reconstruction_loss)
	loss = alpha *  denoising_loss + beta * reconstruction_loss
	tf.scalar_summary('losses/Total', loss)
	return net, loss

def loss(logits, y):
	#slim.losses.softmax_cross_entropy(logits, y)
	slim.losses.mean_squared_error(logits, y)
	#slim.losses.mean_pairwise_squared_error(logits, y)
	total_loss = slim.losses.get_total_loss()
	tf.scalar_summary('losses/Total Loss', total_loss)
	return total_loss

def train(loss, lr = 7):
	optimizer = tf.train.GradientDescentOptimizer(lr)
	#optimizer = tf.train.AdamOptimizer()
	train_op = slim.learning.create_train_op(loss, optimizer)
	return train_op

def error_metrics(logits, y):
	return slim.metrics.aggregate_metric_map({
			'eval/mae': slim.metrics.streaming_mean_absolute_error(logits, y),
			'eval/rmse': slim.metrics.streaming_root_mean_squared_error(logits, y),
		})