from dataset import load_movielens_100k, load_movielens_1m
import model
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		
		users = 6040
		num_items = 3706

		trainset, testset = load_movielens_1m()
		fullset = trainset + testset
		X = tf.placeholder(tf.float32, [None, num_items])
		logits, _ = model.AE(X, is_training=False)
		predict = tf.nn.top_k(logits, 25)

		restorer = tf.train.Saver()
		with tf.Session() as sess:
			# Restore variables from disk.
			path = tf.train.latest_checkpoint('trainAE.log')
			restorer.restore(sess, path)
			print("Model restored.")
			prediction = sess.run(predict, feed_dict={X:trainset[[1, 2553, 4232]]})

			#removing data from trainset
			print prediction
			