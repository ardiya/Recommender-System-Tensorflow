import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def convert_to_tensors(dataset, noise=0):
	X = tf.constant(dataset + np.random.normal(0, 0.2, dataset.shape),
		dtype=tf.float32)
	X.set_shape([None, dataset.shape[1]])
	y = tf.constant(dataset, dtype=tf.float32)
	y.set_shape([None, dataset.shape[1]])
	return X, y

def AE(X, layers=[100, 50, 20], is_training=True):
	users, items = X.get_shape()
	layers = layers + layers[-2::-1]
	print("layers", layers)
	net = X
	for hidden_unit in layers:
		net = slim.fully_connected(net, hidden_unit)
		net = slim.dropout(net, 0.5, is_training=is_training)
	net = slim.fully_connected(net, int(items), tf.nn.sigmoid)
	return net

def loss(logits, y):
	slim.losses.softmax_cross_entropy(logits, y)
	total_loss = slim.losses.get_total_loss()
	tf.scalar_summary('losses/Total Loss', total_loss)
	return total_loss

def train(loss):
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
	train_op = slim.learning.create_train_op(loss, optimizer)
	return train_op

def error_metrics(logits, y):
	return slim.metrics.aggregate_metric_map({
			'eval/mae': slim.metrics.streaming_mean_absolute_error(logits, y),
			'eval/rmse': slim.metrics.streaming_root_mean_squared_error(logits, y),
		})