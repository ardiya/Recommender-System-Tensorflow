from dataset import load_movielens_100k, load_movielens_1m
import model
import tensorflow as tf
slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		
		users = 6040
		num_items = 3706

		trainset, testset = load_movielens_1m()
		X = tf.placeholder(tf.float32, [None, num_items])
		logits = model.AE(X, is_training=False)
		predict = tf.nn.top_k(logits, 25)

		restorer = tf.train.Saver()
		with tf.Session() as sess:
			# Restore variables from disk.
			restorer.restore(sess, "trainAE.log/model.ckpt-1000")
			print("Model restored.")
			prediction = sess.run(predict, feed_dict={X:testset[[1238, 2523, 4232]]})

			print prediction
